"""Causal, inference-safe telemetry adapters for StateBundle.

The simulator intentionally keeps oracle state (for example injected faults)
outside this module.  This recorder receives operational state and observable
events only, retains the complete history for one episode, and constructs the
canonical observation tuple:

    (observation_id, channel, window, payload, metadata)

The public snapshot is suitable for both StateBundle and non-StateBundle
methods.  Evaluator labels remain in a separate privileged
module and are never accepted as input here.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import hmac
import json
import math
import re
import secrets
import statistics
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = "statebundle.canonical.v1"
CHANNELS = ("log", "metric", "alert", "trace", "config")
CHANNEL_ORDER = {channel: index for index, channel in enumerate(CHANNELS)}
REQUIRED_PAYLOAD_FIELDS = {
    "log": {
        "unit_type",
        "template_id",
        "template",
        "event_type",
        "count",
        "severity",
        "severity_histogram",
        "rarity",
        "burst_rate_per_minute",
        "variable_summaries",
        "time_features",
    },
    "metric": {
        "unit_type",
        "metric_name",
        "unit",
        "scale",
        "sample_period_seconds",
        "timestamps_seconds",
        "values",
        "missingness_mask",
        "statistics",
        "normalization_reference",
        "resource",
    },
    "alert": {
        "unit_type",
        "alert_fingerprint",
        "alert_type",
        "message",
        "target",
        "status",
        "severity",
        "threshold",
        "duration_seconds",
        "details",
    },
    "trace": {
        "unit_type",
        "operation",
        "source",
        "destination",
        "count",
        "status_counts",
        "retry_count",
        "latency_ms",
        "critical_path",
    },
    "config": {
        "unit_type",
        "path",
        "value",
        "value_type",
        "previous_value",
        "scope",
        "operation",
        "change_time_seconds",
    },
}
SNAPSHOT_FIELDS = {
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
WINDOW_FIELDS = {
    "start_time_seconds",
    "end_time_seconds",
    "start_inclusive",
    "end_inclusive",
}
OBSERVATION_FIELDS = {
    "observation_id",
    "channel",
    "window",
    "payload",
    "metadata",
}
METADATA_FIELDS = {
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
DATA_QUALITY_FIELDS = {
    "parse_confidence",
    "missingness_fraction",
    "delay_seconds",
    "availability_mask",
    "validation_flags",
}
POLICY_FIELDS = {
    "availability_predicate",
    "host_visibility",
    "timestamp_clock",
    "metric_cadence_seconds",
    "adapter_pipeline",
    "label_plane",
}
ENTITY_ROLES = frozenset({"producer", "target", "source", "destination", "scope"})
HIDDEN_EVENT_TYPES = frozenset({"fault_injected", "fault_expired", "fault_removed"})
HIDDEN_ALERT_TYPES = frozenset({"active_fault"})
FAULT_MECHANISMS = frozenset(
    {
        "application_error",
        "autoscaler_misconfiguration",
        "control_plane_degradation",
        "cooling_degradation",
        "intermittent_server_failure",
        "load_balancer_misconfiguration",
        "monitoring_pipeline_failure",
        "network_congestion_burst",
        "network_partition",
        "placement_policy_misconfiguration",
        "power_budget_violation",
        "power_overload",
        "rack_hotspot",
        "server_failure",
        "storage_io_saturation",
        "thermal_sensor_miscalibration",
        "thermal_throttling",
        "tor_packet_loss",
    }
)
FORBIDDEN_INFERENCE_KEYS = frozenset(
    {
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
)
_UNSAFE_DETAIL_KEY_PARTS = (
    "allocated_server",
    "allocation_weights",
    "backend_weights",
    "fault",
    "maintenance_affected_server",
    "server_id",
    "desired",
    "ground_truth",
    "scenario",
    "seed",
    "propagation",
)
_UNSAFE_DETAIL_EXACT_KEYS = frozenset(
    {
        "base_latency_ms",
        "baseline_capacity_kw",
        "cpu_cost_per_request",
        "latency_fault_penalty_ms",
        "memory_cost_per_request_mb",
        "network_base_latency_ms",
        "network_capacity_mbps",
        "network_congestion_penalty_ms",
        "network_kb_per_request",
        "noise_enabled",
        "noise_stddev",
        "service_latency_ms",
        "storage_base_latency_ms",
        "storage_capacity_iops",
        "storage_congestion_penalty_ms",
        "storage_io_per_request",
    }
)
_SEVERITY_ORDER = {"debug": 0, "info": 1, "warning": 2, "critical": 3}
_MAX_FRAMES_PER_EPISODE = 100_000
_EXACT_SERVER_IDENTIFIER = re.compile(
    r"(?<![A-Za-z0-9_])server-(r\d+)-(row\d+)-rack(\d+)-\d+"
    r"(?![A-Za-z0-9_])",
    flags=re.IGNORECASE,
)


class CanonicalTelemetryError(ValueError):
    """Raised when telemetry violates the canonical inference contract."""


@dataclass(frozen=True)
class MetricDefinition:
    name: str
    unit: str
    subsystem: str
    scale: str = "linear"


METRIC_DEFINITIONS: dict[str, MetricDefinition] = {
    "facility.total_it_power": MetricDefinition(
        "facility.total_it_power", "kW", "power"
    ),
    "facility.total_cooling_power": MetricDefinition(
        "facility.total_cooling_power", "kW", "thermal"
    ),
    "facility.total_power": MetricDefinition("facility.total_power", "kW", "power"),
    "facility.pue": MetricDefinition("facility.pue", "ratio", "facility"),
    "rack.inlet_temperature": MetricDefinition(
        "rack.inlet_temperature", "degC", "thermal"
    ),
    "rack.outlet_temperature": MetricDefinition(
        "rack.outlet_temperature", "degC", "thermal"
    ),
    "rack.power": MetricDefinition("rack.power", "kW", "power"),
    "rack.cpu_utilization": MetricDefinition(
        "rack.cpu_utilization", "percent", "compute"
    ),
    "rack.failed_server_count": MetricDefinition(
        "rack.failed_server_count", "count", "hardware"
    ),
    "rack.server_temperature_mean": MetricDefinition(
        "rack.server_temperature_mean", "degC", "thermal"
    ),
    "rack.reported_inlet_temperature": MetricDefinition(
        "rack.reported_inlet_temperature", "degC", "thermal"
    ),
    "rack.temperature_sensor_disagreement": MetricDefinition(
        "rack.temperature_sensor_disagreement", "degC", "thermal"
    ),
    "rack.temperature_sensor_unhealthy": MetricDefinition(
        "rack.temperature_sensor_unhealthy", "count", "thermal"
    ),
    "rack.power_budget_utilization": MetricDefinition(
        "rack.power_budget_utilization", "ratio", "power"
    ),
    "rack.health_flapping_count": MetricDefinition(
        "rack.health_flapping_count", "count", "hardware"
    ),
    "rack.thermal_throttle_factor": MetricDefinition(
        "rack.thermal_throttle_factor", "ratio", "compute"
    ),
    "rack.network_packet_loss": MetricDefinition(
        "rack.network_packet_loss", "percent", "network"
    ),
    "rack.network_retransmit_rate": MetricDefinition(
        "rack.network_retransmit_rate", "events/second", "network"
    ),
    "rack.network_error_rate": MetricDefinition(
        "rack.network_error_rate", "percent", "network"
    ),
    "cooling.capacity": MetricDefinition("cooling.capacity", "kW", "cooling"),
    "cooling.supply_air_temperature": MetricDefinition(
        "cooling.supply_air_temperature", "degC", "cooling"
    ),
    "cooling.fan_speed": MetricDefinition("cooling.fan_speed", "percent", "cooling"),
    "control_plane.scheduler_api_latency": MetricDefinition(
        "control_plane.scheduler_api_latency", "milliseconds", "control_plane"
    ),
    "control_plane.scheduler_pending_operations": MetricDefinition(
        "control_plane.scheduler_pending_operations", "count", "control_plane"
    ),
    "workload.current_demand": MetricDefinition(
        "workload.current_demand", "requests/second", "workload"
    ),
    "workload.configured_demand": MetricDefinition(
        "workload.configured_demand", "requests/second", "workload"
    ),
    "workload.queue_length": MetricDefinition(
        "workload.queue_length", "requests", "workload"
    ),
    "workload.service_capacity": MetricDefinition(
        "workload.service_capacity", "requests/second", "workload"
    ),
    "workload.average_latency": MetricDefinition(
        "workload.average_latency", "milliseconds", "application"
    ),
    "workload.p95_latency": MetricDefinition(
        "workload.p95_latency", "milliseconds", "application"
    ),
    "workload.network_demand": MetricDefinition(
        "workload.network_demand", "megabits/second", "network"
    ),
    "workload.network_congestion": MetricDefinition(
        "workload.network_congestion", "ratio", "network"
    ),
    "workload.network_packet_loss": MetricDefinition(
        "workload.network_packet_loss", "percent", "network"
    ),
    "workload.network_retransmit_rate": MetricDefinition(
        "workload.network_retransmit_rate", "events/second", "network"
    ),
    "workload.network_error_rate": MetricDefinition(
        "workload.network_error_rate", "percent", "network"
    ),
    "workload.storage_demand": MetricDefinition(
        "workload.storage_demand", "operations/second", "storage"
    ),
    "workload.storage_utilization": MetricDefinition(
        "workload.storage_utilization", "ratio", "storage"
    ),
    "workload.error_rate": MetricDefinition(
        "workload.error_rate", "percent", "application"
    ),
    "workload.dropped_requests": MetricDefinition(
        "workload.dropped_requests", "requests/second", "application"
    ),
    "workload.gpu_utilization": MetricDefinition(
        "workload.gpu_utilization", "percent", "compute"
    ),
    "control_plane.autoscaler_effective_server_limit": MetricDefinition(
        "control_plane.autoscaler_effective_server_limit", "count", "control_plane"
    ),
    "control_plane.autoscaler_target_utilization": MetricDefinition(
        "control_plane.autoscaler_target_utilization", "percent", "control_plane"
    ),
    "control_plane.autoscaler_max_capacity": MetricDefinition(
        "control_plane.autoscaler_max_capacity", "count", "control_plane"
    ),
    "control_plane.autoscaler_cooldown": MetricDefinition(
        "control_plane.autoscaler_cooldown", "seconds", "control_plane"
    ),
    "control_plane.placement_policy_violations": MetricDefinition(
        "control_plane.placement_policy_violations", "count", "control_plane"
    ),
    "workload.placement_imbalance": MetricDefinition(
        "workload.placement_imbalance", "ratio", "scheduler"
    ),
    "observability.metrics_last_updated": MetricDefinition(
        "observability.metrics_last_updated", "seconds", "observability"
    ),
    "observability.logs_last_updated": MetricDefinition(
        "observability.logs_last_updated", "seconds", "observability"
    ),
    "observability.telemetry_lag": MetricDefinition(
        "observability.telemetry_lag", "seconds", "observability"
    ),
    "observability.metrics_missing_ratio": MetricDefinition(
        "observability.metrics_missing_ratio", "ratio", "observability"
    ),
    "observability.logs_missing_ratio": MetricDefinition(
        "observability.logs_missing_ratio", "ratio", "observability"
    ),
    "application.load_balancer_backend_skew": MetricDefinition(
        "application.load_balancer_backend_skew", "ratio", "application"
    ),
    "application.load_balancer_unhealthy_routing": MetricDefinition(
        "application.load_balancer_unhealthy_routing", "ratio", "application"
    ),
    "application.load_balancer_error_rate": MetricDefinition(
        "application.load_balancer_error_rate", "percent", "application"
    ),
}


class CanonicalTelemetryRecorder:
    """Retain one episode's raw observable history and build causal snapshots."""

    def __init__(
        self,
        episode_id: str,
        tick_seconds: int,
        *,
        opaque_key: bytes | str | None = None,
    ) -> None:
        if not episode_id:
            raise CanonicalTelemetryError("episode_id must be non-empty")
        if type(tick_seconds) is not int or tick_seconds < 1:
            raise CanonicalTelemetryError("tick_seconds must be a positive integer")
        self.episode_id = str(episode_id)
        self.tick_seconds = tick_seconds
        if opaque_key is None:
            self._opaque_key = secrets.token_bytes(32)
        elif isinstance(opaque_key, str):
            self._opaque_key = opaque_key.encode("utf-8")
        elif isinstance(opaque_key, bytes):
            self._opaque_key = opaque_key
        else:
            raise CanonicalTelemetryError("opaque_key must be bytes or text")
        if len(self._opaque_key) < 16:
            raise CanonicalTelemetryError("opaque_key must contain at least 16 bytes")
        self._record_sequence = 0
        self._capture_sequence = 0
        self._frames: list[dict[str, Any]] = []
        self._logs: list[dict[str, Any]] = []
        initial_token = self._cut_token(0)
        self._cut_sequence_by_token = {initial_token: 0}

    def _opaque_digest(self, namespace: str, value: Any, *, length: int) -> str:
        serialized = json.dumps(
            {
                "episode_id": self.episode_id,
                "namespace": namespace,
                "value": value,
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            default=str,
        ).encode("utf-8")
        return hmac.new(self._opaque_key, serialized, hashlib.sha256).hexdigest()[
            :length
        ]

    def _cut_token(self, sequence: int) -> str:
        token = "cut-" + self._opaque_digest(
            "causal-cut",
            {"private_sequence": sequence},
            length=32,
        )
        if hasattr(self, "_cut_sequence_by_token"):
            self._cut_sequence_by_token[token] = sequence
        return token

    def record_event(
        self,
        event: Mapping[str, Any],
        *,
        available_at_time_seconds: int | float | None = None,
    ) -> None:
        """Record an observable log event.

        Oracle lifecycle events are discarded, and event details are recursively
        reduced to source-observable fields before being retained.
        """
        event_type = str(event.get("event_type", "unknown"))
        if event_type in HIDDEN_EVENT_TYPES:
            return
        event_time = _number(event.get("sim_time_seconds"), default=0.0)
        available_at = (
            event_time
            if available_at_time_seconds is None
            else _number(available_at_time_seconds, default=event_time)
        )
        if available_at < event_time:
            raise CanonicalTelemetryError(
                "available_at_time_seconds precedes event time"
            )
        message = str(event.get("message", event_type))
        if _contains_fault_mechanism(message):
            return
        safe_details = _safe_details(event.get("details", {}))
        source_sequence = int(event.get("sequence_id", len(self._logs) + 1))
        self._record_sequence += 1
        self._logs.append(
            {
                "source_sequence": source_sequence,
                "available_at_sequence": self._record_sequence,
                "event_type": event_type,
                "event_time_seconds": event_time,
                "available_at_time_seconds": available_at,
                "message": message,
                "details": safe_details,
                "severity": _event_severity(event_type),
            }
        )

    def capture(
        self,
        simulator: Any,
        *,
        reason: str,
        available_at_time_seconds: int | float | None = None,
    ) -> None:
        """Capture one operational state frame after a simulator transition."""
        event_time = _number(simulator.sim_time_seconds, default=0.0)
        available_at = (
            event_time
            if available_at_time_seconds is None
            else _number(available_at_time_seconds, default=event_time)
        )
        if available_at < event_time:
            raise CanonicalTelemetryError(
                "available_at_time_seconds precedes event time"
            )
        if len(self._frames) >= _MAX_FRAMES_PER_EPISODE:
            raise CanonicalTelemetryError(
                "canonical frame limit reached; stream the episode before continuing"
            )

        self._record_sequence += 1
        self._capture_sequence += 1
        metrics = _extract_metric_samples(simulator)
        self._annotate_monitoring_metric_delivery(
            metrics,
            event_time=event_time,
            available_at_time=available_at,
        )
        traces = _extract_trace_samples(simulator, self.tick_seconds)
        safe_alerts = [
            *_extract_safe_alerts(simulator),
            *self._derive_observable_alerts(metrics, event_time, available_at),
        ]
        safe_config = _extract_safe_configuration(simulator)
        self._frames.append(
            {
                "capture_sequence": self._capture_sequence,
                "available_at_sequence": self._record_sequence,
                "reason": str(reason),
                "event_time_seconds": event_time,
                "available_at_time_seconds": available_at,
                "metrics": metrics,
                "traces": traces,
                "alerts": safe_alerts,
                "config": safe_config,
            }
        )

    def _derive_observable_alerts(
        self,
        current_metrics: Sequence[dict[str, Any]],
        event_time: float,
        available_at: float,
    ) -> list[dict[str, Any]]:
        """Generate source-observable threshold alerts without oracle state.

        Some simulator effects (placement loss, reduced service capacity, and
        storage saturation) do not cross the legacy alert thresholds.  These
        rules compare current operational measurements with the immediately
        preceding causal frame.  They never inspect active faults.
        """
        eligible_frames = [
            frame
            for frame in self._frames
            if frame["event_time_seconds"] <= event_time
            if frame["available_at_time_seconds"] <= available_at
        ]
        if not eligible_frames:
            return []
        previous_frame = max(
            eligible_frames,
            key=lambda frame: (
                frame["event_time_seconds"],
                frame["capture_sequence"],
            ),
        )
        previous_metrics = {
            (item["name"], item["entity_id"]): item["value"]
            for item in previous_frame["metrics"]
        }
        current = {
            (item["name"], item["entity_id"]): item["value"] for item in current_metrics
        }
        alerts: list[dict[str, Any]] = []
        demand = current.get(("workload.current_demand", "workload"), 0.0)

        def active_alert(alert_type: str, target: str) -> dict[str, Any] | None:
            return next(
                (
                    alert
                    for alert in previous_frame.get("alerts", [])
                    if alert.get("alert_type") == alert_type
                    and alert.get("target") == target
                ),
                None,
            )

        if demand > 0:
            for (metric_name, entity_id), value in sorted(current.items()):
                if metric_name != "rack.cpu_utilization":
                    continue
                previous = previous_metrics.get((metric_name, entity_id), 0.0)
                active = active_alert("RackTrafficDrop", entity_id)
                active_details = (
                    active.get("details", {}) if isinstance(active, Mapping) else {}
                )
                reference = float(
                    active_details.get("reference_cpu_utilization_percent", previous)
                )
                if reference > 0.05 and value <= reference * (0.5 if active else 0.1):
                    alerts.append(
                        {
                            "alert_type": "RackTrafficDrop",
                            "severity": "warning",
                            "target": entity_id,
                            "message": "Rack workload traffic dropped unexpectedly",
                            "details": {
                                "cpu_utilization_percent": value,
                                "reference_cpu_utilization_percent": reference,
                                "drop_ratio_threshold": 0.9,
                            },
                        }
                    )

        for (metric_name, entity_id), value in sorted(current.items()):
            if metric_name != "cooling.capacity":
                continue
            previous = previous_metrics.get((metric_name, entity_id), 0.0)
            active = active_alert("CoolingCapacityDrop", entity_id)
            active_details = (
                active.get("details", {}) if isinstance(active, Mapping) else {}
            )
            reference = float(
                active_details.get("reference_cooling_capacity_kw", previous)
            )
            if reference > 0 and value <= reference * (0.9 if active else 0.8):
                alerts.append(
                    {
                        "alert_type": "CoolingCapacityDrop",
                        "severity": "critical",
                        "target": entity_id,
                        "message": "Cooling-unit capacity dropped below its recent level",
                        "details": {
                            "cooling_capacity_kw": value,
                            "reference_cooling_capacity_kw": reference,
                            "drop_ratio_threshold": 0.2,
                        },
                    }
                )

        capacity_key = ("workload.service_capacity", "workload")
        previous_capacity = previous_metrics.get(capacity_key, 0.0)
        current_capacity = current.get(capacity_key, 0.0)
        active_capacity = active_alert("ServiceCapacityDrop", "workload")
        active_capacity_details = (
            active_capacity.get("details", {})
            if isinstance(active_capacity, Mapping)
            else {}
        )
        reference_capacity = float(
            active_capacity_details.get(
                "reference_service_capacity_requests_per_second",
                previous_capacity,
            )
        )
        if (
            demand > 0
            and reference_capacity > 0
            and current_capacity
            < reference_capacity * (0.8 if active_capacity else 0.6)
        ):
            alerts.append(
                {
                    "alert_type": "ServiceCapacityDrop",
                    "severity": "critical",
                    "target": "workload",
                    "message": "Observable workload service capacity dropped sharply",
                    "details": {
                        "service_capacity_requests_per_second": current_capacity,
                        "reference_service_capacity_requests_per_second": reference_capacity,
                        "drop_ratio_threshold": 0.4,
                    },
                }
            )

        storage_key = ("workload.storage_utilization", "workload")
        previous_storage = previous_metrics.get(storage_key, 0.0)
        current_storage = current.get(storage_key, 0.0)
        active_storage = active_alert("StorageUtilizationJump", "workload")
        active_storage_details = (
            active_storage.get("details", {})
            if isinstance(active_storage, Mapping)
            else {}
        )
        reference_storage = float(
            active_storage_details.get(
                "reference_storage_utilization_ratio",
                previous_storage,
            )
        )
        if (
            demand > 0
            and reference_storage > 0
            and current_storage >= 0.02
            and current_storage > reference_storage * (2.0 if active_storage else 4.0)
        ):
            alerts.append(
                {
                    "alert_type": "StorageUtilizationJump",
                    "severity": "warning",
                    "target": "workload",
                    "message": "Storage utilization increased sharply",
                    "details": {
                        "storage_utilization_ratio": current_storage,
                        "reference_storage_utilization_ratio": reference_storage,
                        "increase_ratio_threshold": 4.0,
                    },
                }
            )

        latency_key = ("workload.average_latency", "workload")
        previous_latency = previous_metrics.get(latency_key, 0.0)
        current_latency = current.get(latency_key, 0.0)
        active_latency = active_alert("LatencyStepIncrease", "workload")
        active_latency_details = (
            active_latency.get("details", {})
            if isinstance(active_latency, Mapping)
            else {}
        )
        reference_latency = float(
            active_latency_details.get("reference_average_latency_ms", previous_latency)
        )
        if (
            demand > 0
            and reference_latency > 0
            and current_latency >= 20.0
            and current_latency > reference_latency * (1.5 if active_latency else 2.0)
        ):
            alerts.append(
                {
                    "alert_type": "LatencyStepIncrease",
                    "severity": "warning",
                    "target": "workload",
                    "message": "Request latency increased sharply",
                    "details": {
                        "average_latency_ms": current_latency,
                        "reference_average_latency_ms": reference_latency,
                        "increase_ratio_threshold": 2.0,
                    },
                }
            )

        scheduler_latency = current.get(
            ("control_plane.scheduler_api_latency", "control-plane"),
            0.0,
        )
        if scheduler_latency >= 20.0:
            alerts.append(
                {
                    "alert_type": "SchedulerAPILatencyHigh",
                    "severity": "warning",
                    "target": "control-plane",
                    "message": "Scheduler API latency is elevated",
                    "details": {
                        "scheduler_api_latency_ms": scheduler_latency,
                        "threshold_ms": 20.0,
                    },
                }
            )
        return alerts

    def _annotate_monitoring_metric_delivery(
        self,
        metrics: list[dict[str, Any]],
        *,
        event_time: float,
        available_at_time: float,
    ) -> None:
        """Persist delivery semantics for samples captured during an outage.

        Query-time staleness controls what is visible while the pipeline is
        impaired. These private annotations additionally prevent repair from
        making outage-period samples appear retroactively. Delayed samples can
        become visible after their modeled delivery time; dropped samples
        remain missing. The independent ``observability.*`` health stream is
        deliberately exempt.
        """
        observability = {
            sample["name"]: sample["value"]
            for sample in metrics
            if sample["entity_id"] == "monitoring-pipeline"
        }
        lag_seconds = max(
            0.0,
            float(observability.get("observability.telemetry_lag", 0.0)),
        )
        missing_ratio = min(
            1.0,
            max(
                0.0,
                float(
                    observability.get(
                        "observability.metrics_missing_ratio",
                        0.0,
                    )
                ),
            ),
        )
        if lag_seconds <= 0.0 and missing_ratio <= 0.0:
            return
        for sample in metrics:
            if sample["name"].startswith("observability."):
                continue
            sample["_pipeline_available_at_time_seconds"] = (
                max(event_time, available_at_time) + lag_seconds
            )
            sample["_pipeline_dropped"] = not self._monitoring_sample_retained(
                "metric-delivery",
                (
                    sample["name"],
                    sample["entity_id"],
                    event_time,
                    self._capture_sequence,
                ),
                missing_ratio,
            )

    def snapshot(
        self,
        *,
        query_time_seconds: int | float,
        query_watermark_sequence: str | None = None,
        lookback_seconds: int | float = 300,
        channels: Iterable[str] | None = None,
        include_config: bool = True,
        log_limit: int | None = None,
    ) -> dict[str, Any]:
        """Return an immutable canonical snapshot with no future information.

        Simulation time is not a total order because several transitions may
        happen within one tick. ``query_watermark_sequence`` makes the causal
        cut replayable without exposing simulator ground truth.
        """
        query_time = _number(query_time_seconds)
        if query_watermark_sequence is None:
            query_watermark_private = max(
                (
                    item["available_at_sequence"]
                    for item in [*self._logs, *self._frames]
                    if item["event_time_seconds"] <= query_time
                    and item["available_at_time_seconds"] <= query_time
                ),
                default=0,
            )
        elif (
            not isinstance(query_watermark_sequence, str)
            or query_watermark_sequence not in self._cut_sequence_by_token
        ):
            raise CanonicalTelemetryError(
                "query_watermark_sequence must be an opaque cut token issued "
                "by this episode"
            )
        else:
            query_watermark_private = self._cut_sequence_by_token[
                query_watermark_sequence
            ]
        query_watermark_token = self._cut_token(query_watermark_private)
        monitoring_policy = self._monitoring_pipeline_policy(
            query_time,
            query_watermark_private,
        )
        lookback = _number(lookback_seconds)
        if lookback < 0:
            raise CanonicalTelemetryError("lookback_seconds must be non-negative")
        if not isinstance(include_config, bool):
            raise CanonicalTelemetryError("include_config must be a boolean")
        if log_limit is not None and (
            isinstance(log_limit, bool)
            or not isinstance(log_limit, int)
            or log_limit < 0
        ):
            raise CanonicalTelemetryError(
                "log_limit must be a non-negative integer or null"
            )
        requested_channels = tuple(
            channel
            for channel in _normalize_channels(channels)
            if include_config or channel != "config"
        )
        window_start = max(0.0, query_time - lookback)

        observations: list[dict[str, Any]] = []
        if "log" in requested_channels:
            log_observations = self._adapt_logs(
                window_start,
                query_time,
                query_watermark_private,
                monitoring_policy,
            )
            if log_limit is not None:
                log_observations = sorted(
                    log_observations,
                    key=lambda item: (
                        item["metadata"]["event_end_time_seconds"],
                        item["observation_id"],
                    ),
                    reverse=True,
                )[:log_limit]
            observations.extend(log_observations)
        if "metric" in requested_channels:
            observations.extend(
                self._adapt_metrics(
                    window_start,
                    query_time,
                    query_watermark_private,
                    monitoring_policy,
                )
            )
        if "alert" in requested_channels:
            observations.extend(
                self._adapt_alerts(
                    window_start,
                    query_time,
                    query_watermark_private,
                )
            )
        if "trace" in requested_channels:
            observations.extend(
                self._adapt_traces(
                    window_start,
                    query_time,
                    query_watermark_private,
                )
            )
        if "config" in requested_channels:
            observations.extend(
                self._adapt_config(
                    window_start,
                    query_time,
                    query_watermark_private,
                )
            )

        # A public record ordinal would reveal hidden lifecycle transitions.
        # Every emitted observation therefore carries only the opaque token
        # attesting that it belongs to this causal cut.
        for observation in observations:
            observation["metadata"]["available_at_sequence"] = query_watermark_token
        observations.sort(
            key=lambda item: (
                CHANNEL_ORDER[item["channel"]],
                item["window"]["start_time_seconds"],
                item["observation_id"],
            )
        )
        body = {
            "schema_version": SCHEMA_VERSION,
            "episode_id": self.episode_id,
            "query_time_seconds": query_time,
            "query_watermark_sequence": query_watermark_token,
            "window": {
                "start_time_seconds": window_start,
                "end_time_seconds": query_time,
                "start_inclusive": True,
                "end_inclusive": True,
            },
            "channels": list(requested_channels),
            "channel_counts": {
                channel: sum(item["channel"] == channel for item in observations)
                for channel in requested_channels
            },
            "policy": {
                "availability_predicate": (
                    "available_at_time_seconds <= query_time_seconds and "
                    "available_at_sequence == query_watermark_sequence"
                ),
                "host_visibility": "rack",
                "timestamp_clock": "simulation_seconds",
                "metric_cadence_seconds": self.tick_seconds,
                "adapter_pipeline": [
                    "decode",
                    "consolidate",
                    "normalize",
                    "window",
                    "validate",
                ],
                "label_plane": "separate_training_sidecar",
            },
            "observations": observations,
        }
        body["snapshot_id"] = "snapshot-" + _stable_hash(body, length=24)
        assert_inference_safe(body)
        return deepcopy(body)

    def _monitoring_pipeline_policy(
        self,
        query_time: float,
        query_watermark: int,
    ) -> dict[str, float]:
        policy = {
            "metrics_last_updated": query_time,
            "logs_last_updated": query_time,
            "metrics_missing_ratio": 0.0,
            "logs_missing_ratio": 0.0,
        }
        eligible_frames = [
            frame
            for frame in self._frames
            if frame["event_time_seconds"] <= query_time
            and frame["available_at_time_seconds"] <= query_time
            and frame["available_at_sequence"] <= query_watermark
        ]
        if not eligible_frames:
            return policy
        latest_frame = max(
            eligible_frames,
            key=lambda frame: (
                frame["event_time_seconds"],
                frame["capture_sequence"],
            ),
        )
        values = {
            sample["name"]: sample["value"]
            for sample in latest_frame["metrics"]
            if sample["entity_id"] == "monitoring-pipeline"
        }
        telemetry_lag = float(values.get("observability.telemetry_lag", 0.0))
        if telemetry_lag <= 0:
            return policy
        policy["metrics_last_updated"] = min(
            query_time,
            max(
                0.0,
                float(
                    values.get(
                        "observability.metrics_last_updated",
                        query_time - telemetry_lag,
                    )
                ),
            ),
        )
        policy["logs_last_updated"] = min(
            query_time,
            max(
                0.0,
                float(
                    values.get(
                        "observability.logs_last_updated",
                        query_time - telemetry_lag,
                    )
                ),
            ),
        )
        policy["metrics_missing_ratio"] = min(
            1.0,
            max(
                0.0,
                float(
                    values.get(
                        "observability.metrics_missing_ratio",
                        0.0,
                    )
                ),
            ),
        )
        policy["logs_missing_ratio"] = min(
            1.0,
            max(
                0.0,
                float(
                    values.get(
                        "observability.logs_missing_ratio",
                        0.0,
                    )
                ),
            ),
        )
        return policy

    def _adapt_logs(
        self,
        window_start: float,
        query_time: float,
        query_watermark: int,
        monitoring_policy: Mapping[str, float],
    ) -> list[dict[str, Any]]:
        log_cutoff = monitoring_policy["logs_last_updated"]
        available = [
            item
            for item in self._logs
            if window_start <= item["event_time_seconds"] <= min(query_time, log_cutoff)
            and item["available_at_time_seconds"] <= query_time
            and item["available_at_sequence"] <= query_watermark
        ]
        missing_ratio = monitoring_policy["logs_missing_ratio"]
        if missing_ratio > 0.0 and available:
            retained = [
                item
                for item in available
                if self._monitoring_sample_retained(
                    "log",
                    (
                        item["source_sequence"],
                        item["event_type"],
                        item["event_time_seconds"],
                    ),
                    missing_ratio,
                )
            ]
            available = retained or [available[0]]
        grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        if log_cutoff < query_time or missing_ratio > 0.0:
            # Count only events surviving the outage policy. Otherwise rarity
            # would disclose fresh events that the broken log path did not
            # deliver.
            causal_counts: Counter[str] = Counter(
                _log_template_id(item) for item in available
            )
        else:
            # Preserve the healthy-path causal rarity definition, including
            # events before this query's display window.
            causal_counts = Counter(
                _log_template_id(item)
                for item in self._logs
                if item["available_at_time_seconds"] <= query_time
                and item["available_at_sequence"] <= query_watermark
            )
        for item in available:
            subsystem = _subsystem_for_log(item)
            grouped[(subsystem, _log_template_id(item))].append(item)

        observations = []
        for (subsystem, template_id), events in sorted(grouped.items()):
            start = min(item["event_time_seconds"] for item in events)
            end = max(item["event_time_seconds"] for item in events)
            severity_histogram = dict(
                sorted(Counter(item["severity"] for item in events).items())
            )
            details = [item["details"] for item in events if item["details"]]
            payload = {
                "unit_type": "template_aggregate",
                "template_id": template_id,
                "template": _normalize_template(events[0]["message"]),
                "event_type": events[0]["event_type"],
                "count": len(events),
                "severity": max(
                    (item["severity"] for item in events),
                    key=lambda item: _SEVERITY_ORDER.get(item, 0),
                ),
                "severity_histogram": severity_histogram,
                "rarity": round(1.0 / max(1, causal_counts[template_id]), 6),
                "burst_rate_per_minute": round(
                    len(events)
                    * 60.0
                    / max(self.tick_seconds, end - start + self.tick_seconds),
                    6,
                ),
                "variable_summaries": _variable_summaries(details),
                "time_features": {
                    "first_offset_seconds": round(start - window_start, 6),
                    "last_offset_seconds": round(end - window_start, 6),
                },
            }
            source_sequences = [item["source_sequence"] for item in events]
            observations.append(
                self._observation(
                    channel="log",
                    key=f"{subsystem}:{template_id}",
                    start=start,
                    end=end,
                    available_at=max(
                        item["available_at_time_seconds"] for item in events
                    ),
                    available_at_sequence=max(
                        item["available_at_sequence"] for item in events
                    ),
                    payload=payload,
                    entities=[
                        _entity(subsystem, "producer", provenance="source_event_type")
                    ],
                    subsystem=subsystem,
                    correlations={"log_template_id": template_id},
                    source_keys=[f"log:{sequence}" for sequence in source_sequences],
                    missingness=missing_ratio,
                    availability_mask={
                        "template": True,
                        "severity": True,
                        "rarity": True,
                        "burst": True,
                        "variable_summaries": bool(details),
                    },
                )
            )
        return observations

    def _adapt_metrics(
        self,
        window_start: float,
        query_time: float,
        query_watermark: int,
        monitoring_policy: Mapping[str, float],
    ) -> list[dict[str, Any]]:
        causal_series_and_time: dict[
            tuple[str, str],
            dict[float, dict[str, Any]],
        ] = defaultdict(dict)
        for frame in self._frames:
            event_time = frame["event_time_seconds"]
            if event_time > query_time:
                continue
            if frame["available_at_time_seconds"] > query_time:
                continue
            if frame["available_at_sequence"] > query_watermark:
                continue
            for sample in frame["metrics"]:
                pipeline_available_at = float(
                    sample.get(
                        "_pipeline_available_at_time_seconds",
                        frame["available_at_time_seconds"],
                    )
                )
                if not sample["name"].startswith("observability.") and (
                    bool(sample.get("_pipeline_dropped", False))
                    or pipeline_available_at > query_time
                ):
                    continue
                if (
                    not sample["name"].startswith("observability.")
                    and event_time > monitoring_policy["metrics_last_updated"]
                ):
                    continue
                key = (sample["name"], sample["entity_id"])
                previous = causal_series_and_time[key].get(event_time)
                if (
                    previous is None
                    or frame["capture_sequence"] > previous["capture_sequence"]
                ):
                    causal_series_and_time[key][event_time] = {
                        **sample,
                        "capture_sequence": frame["capture_sequence"],
                        "available_at_time_seconds": max(
                            frame["available_at_time_seconds"],
                            pipeline_available_at,
                        ),
                        "available_at_sequence": frame["available_at_sequence"],
                    }

        missing_ratio = monitoring_policy["metrics_missing_ratio"]
        if missing_ratio > 0.0:
            for key, samples_by_time in causal_series_and_time.items():
                if key[0].startswith("observability.") or not samples_by_time:
                    continue
                retained = {
                    timestamp: sample
                    for timestamp, sample in samples_by_time.items()
                    if self._monitoring_sample_retained(
                        "metric",
                        (key, timestamp),
                        missing_ratio,
                    )
                }
                if not retained:
                    latest_timestamp = max(samples_by_time)
                    retained[latest_timestamp] = samples_by_time[latest_timestamp]
                samples_by_time.clear()
                samples_by_time.update(retained)

        observations = []
        for (metric_name, entity_id), causal_samples_by_time in sorted(
            causal_series_and_time.items()
        ):
            samples_by_time = {
                timestamp: sample
                for timestamp, sample in causal_samples_by_time.items()
                if window_start <= timestamp <= query_time
            }
            if not samples_by_time:
                continue
            definition = METRIC_DEFINITIONS[metric_name]
            timestamps = _expected_timestamps(
                window_start,
                query_time,
                self.tick_seconds,
                observed_times=samples_by_time,
            )
            values: list[float | None] = []
            missing_mask: list[bool] = []
            available_times: list[float] = []
            for timestamp in timestamps:
                sample = samples_by_time.get(timestamp)
                values.append(None if sample is None else sample["value"])
                missing_mask.append(sample is None)
                if sample is not None:
                    available_times.append(sample["available_at_time_seconds"])
            numeric_values = [value for value in values if value is not None]
            if not numeric_values:
                continue
            stats = _numeric_statistics(numeric_values, timestamps, values)
            historical_samples = [
                sample
                for timestamp, sample in sorted(causal_samples_by_time.items())
                if timestamp < window_start
            ]
            reference_values = [
                sample["value"] for sample in historical_samples
            ] or numeric_values
            latest_reference_sample = (
                historical_samples[-1] if historical_samples else None
            )
            median = statistics.median(reference_values)
            q25, q75 = (
                _percentile(reference_values, 0.25),
                _percentile(reference_values, 0.75),
            )
            series_id = "metric-series-" + _stable_hash(
                {
                    "episode_id": self.episode_id,
                    "name": metric_name,
                    "entity": entity_id,
                },
                length=20,
            )
            payload = {
                "unit_type": "series_segment",
                "metric_name": metric_name,
                "unit": definition.unit,
                "scale": definition.scale,
                "sample_period_seconds": self.tick_seconds,
                "timestamps_seconds": timestamps,
                "values": values,
                "missingness_mask": missing_mask,
                "statistics": stats,
                "normalization_reference": {
                    "method": (
                        "causal_pre_window_robust"
                        if historical_samples
                        else "causal_window_fallback"
                    ),
                    "median": round(float(median), 6),
                    "iqr": round(float(q75 - q25), 6),
                    "last": round(
                        float(
                            latest_reference_sample["value"]
                            if latest_reference_sample is not None
                            else numeric_values[-1]
                        ),
                        6,
                    ),
                    "sample_count": len(reference_values),
                    "reference_end_time_seconds": (
                        max(
                            timestamp
                            for timestamp in causal_samples_by_time
                            if timestamp < window_start
                        )
                        if historical_samples
                        else None
                    ),
                },
                "resource": {"entity_id": entity_id},
            }
            missingness = sum(missing_mask) / max(1, len(missing_mask))
            observations.append(
                self._observation(
                    channel="metric",
                    key=f"{series_id}:{window_start}:{query_time}",
                    start=timestamps[0],
                    end=timestamps[-1],
                    available_at=max(available_times),
                    available_at_sequence=max(
                        sample["available_at_sequence"]
                        for sample in samples_by_time.values()
                    ),
                    payload=payload,
                    entities=[
                        _entity(entity_id, "producer", provenance="metric_resource")
                    ],
                    subsystem=definition.subsystem,
                    correlations={"metric_series_id": series_id},
                    source_keys=[
                        f"metric:{metric_name}:{entity_id}:{timestamp}"
                        for timestamp, missing in zip(
                            timestamps, missing_mask, strict=True
                        )
                        if not missing
                    ],
                    missingness=missingness,
                    availability_mask={
                        "unit": True,
                        "sample_period": True,
                        "missingness_mask": True,
                        "statistics": True,
                        "normalization_reference": True,
                        "resource": True,
                    },
                )
            )
        return observations

    def _monitoring_sample_retained(
        self,
        channel: str,
        identity: Any,
        missing_ratio: float,
    ) -> bool:
        digest = int(
            _stable_hash(
                {
                    "episode_id": self.episode_id,
                    "channel": channel,
                    "identity": identity,
                },
                length=8,
            ),
            16,
        )
        quantile = digest / float(0xFFFFFFFF)
        return quantile >= missing_ratio

    def _adapt_alerts(
        self,
        window_start: float,
        query_time: float,
        query_watermark: int,
    ) -> list[dict[str, Any]]:
        """Build alert episodes from only frames inside the causal cut."""
        latest_frame_by_time: dict[float, dict[str, Any]] = {}
        for frame in self._frames:
            if (
                frame["event_time_seconds"] > query_time
                or frame["available_at_time_seconds"] > query_time
                or frame["available_at_sequence"] > query_watermark
            ):
                continue
            previous = latest_frame_by_time.get(frame["event_time_seconds"])
            if (
                previous is None
                or frame["capture_sequence"] > previous["capture_sequence"]
            ):
                latest_frame_by_time[frame["event_time_seconds"]] = frame

        transitions: list[dict[str, Any]] = []
        active: dict[str, dict[str, Any]] = {}
        transition_sequence = 0
        for frame in (
            latest_frame_by_time[event_time]
            for event_time in sorted(latest_frame_by_time)
        ):
            current = {
                _alert_fingerprint(alert): alert for alert in frame.get("alerts", [])
            }
            for fingerprint, alert in sorted(current.items()):
                if fingerprint in active:
                    continue
                transition_sequence += 1
                transitions.append(
                    {
                        "transition_sequence": transition_sequence,
                        "fingerprint": fingerprint,
                        "status": "firing",
                        "alert": alert,
                        "event_time_seconds": frame["event_time_seconds"],
                        "available_at_time_seconds": frame["available_at_time_seconds"],
                        "available_at_sequence": frame["available_at_sequence"],
                    }
                )
            for fingerprint, alert in sorted(active.items()):
                if fingerprint in current:
                    continue
                transition_sequence += 1
                transitions.append(
                    {
                        "transition_sequence": transition_sequence,
                        "fingerprint": fingerprint,
                        "status": "resolved",
                        "alert": alert,
                        "event_time_seconds": frame["event_time_seconds"],
                        "available_at_time_seconds": frame["available_at_time_seconds"],
                        "available_at_sequence": frame["available_at_sequence"],
                    }
                )
            active = current

        by_fingerprint: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for transition in transitions:
            by_fingerprint[transition["fingerprint"]].append(transition)

        observations = []
        for fingerprint, history in sorted(by_fingerprint.items()):
            history.sort(
                key=lambda item: (
                    item["event_time_seconds"],
                    item["transition_sequence"],
                )
            )
            episodes: list[tuple[dict[str, Any], dict[str, Any] | None]] = []
            firing: dict[str, Any] | None = None
            for transition in history:
                if transition["status"] == "firing":
                    if firing is None:
                        firing = transition
                elif firing is not None:
                    episodes.append((firing, transition))
                    firing = None
            if firing is not None:
                episodes.append((firing, None))

            for start_transition, end_transition in episodes:
                episode_start = start_transition["event_time_seconds"]
                episode_end = (
                    query_time
                    if end_transition is None
                    else end_transition["event_time_seconds"]
                )
                if episode_end < window_start or episode_start > query_time:
                    continue
                alert = start_transition["alert"]
                status = "firing" if end_transition is None else "resolved"
                target = str(alert.get("target", "unknown"))
                details = _safe_details(alert.get("details", {}))
                threshold = _alert_threshold(details)
                payload = {
                    "unit_type": "deduplicated_episode",
                    "alert_fingerprint": fingerprint,
                    "alert_type": str(alert.get("alert_type", "unknown")),
                    "message": str(alert.get("message", "")),
                    "target": target,
                    "status": status,
                    "severity": _normalize_severity(alert.get("severity")),
                    "threshold": threshold,
                    "duration_seconds": round(max(0.0, episode_end - episode_start), 6),
                    "details": details,
                }
                observations.append(
                    self._observation(
                        channel="alert",
                        key=f"{fingerprint}:{episode_start}",
                        start=episode_start,
                        end=episode_end,
                        available_at=max(
                            start_transition["available_at_time_seconds"],
                            (
                                end_transition["available_at_time_seconds"]
                                if end_transition is not None
                                else start_transition["available_at_time_seconds"]
                            ),
                        ),
                        available_at_sequence=max(
                            start_transition["available_at_sequence"],
                            (
                                end_transition["available_at_sequence"]
                                if end_transition is not None
                                else start_transition["available_at_sequence"]
                            ),
                        ),
                        payload=payload,
                        entities=[_entity(target, "target", provenance="alert_target")],
                        subsystem=_subsystem_for_alert(alert),
                        correlations={"alert_fingerprint": fingerprint},
                        source_keys=[
                            f"alert:{item['transition_sequence']}"
                            for item in (start_transition, end_transition)
                            if item is not None
                        ],
                        missingness=0.0,
                        availability_mask={
                            "target": target != "unknown",
                            "status": True,
                            "severity": True,
                            "threshold": threshold is not None,
                            "duration": True,
                        },
                    )
                )
        return observations

    def _adapt_traces(
        self,
        window_start: float,
        query_time: float,
        query_watermark: int,
    ) -> list[dict[str, Any]]:
        grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
        for frame in self._frames:
            if not window_start <= frame["event_time_seconds"] <= query_time:
                continue
            if frame["available_at_time_seconds"] > query_time:
                continue
            if frame["available_at_sequence"] > query_watermark:
                continue
            for trace in frame["traces"]:
                grouped[
                    (
                        trace["operation"],
                        trace["source_entity_id"],
                        trace["destination_entity_id"],
                    )
                ].append(
                    {
                        **trace,
                        "event_time_seconds": frame["event_time_seconds"],
                        "available_at_time_seconds": frame["available_at_time_seconds"],
                        "available_at_sequence": frame["available_at_sequence"],
                        "capture_sequence": frame["capture_sequence"],
                    }
                )

        observations = []
        for (operation, source, destination), samples in sorted(grouped.items()):
            by_time: dict[float, dict[str, Any]] = {}
            for sample in samples:
                prior = by_time.get(sample["event_time_seconds"])
                if (
                    prior is None
                    or sample["capture_sequence"] > prior["capture_sequence"]
                ):
                    by_time[sample["event_time_seconds"]] = sample
            samples = [by_time[key] for key in sorted(by_time)]
            start = samples[0]["event_time_seconds"]
            end = samples[-1]["event_time_seconds"]
            total_count = sum(item["count"] for item in samples)
            error_count = sum(item["error_count"] for item in samples)
            retry_count = sum(item["retry_count"] for item in samples)
            latency_values = [
                item["latency_ms"]
                for item in samples
                if item["latency_ms"] is not None and item["count"] > 0
            ]
            p95_values = [
                item["p95_latency_ms"]
                for item in samples
                if item["p95_latency_ms"] is not None and item["count"] > 0
            ]
            group_id = "trace-group-" + _stable_hash(
                {
                    "episode_id": self.episode_id,
                    "operation": operation,
                    "source": source,
                    "destination": destination,
                },
                length=20,
            )
            payload = {
                "unit_type": "source_destination_operation_group",
                "operation": operation,
                "source": source,
                "destination": destination,
                "count": round(total_count, 6),
                "status_counts": {
                    "ok": round(max(0.0, total_count - error_count), 6),
                    "error": round(error_count, 6),
                },
                "retry_count": round(retry_count, 6),
                "latency_ms": {
                    "mean": (
                        round(statistics.fmean(latency_values), 6)
                        if latency_values
                        else None
                    ),
                    "p95": max(p95_values) if p95_values else None,
                    "min": min(latency_values) if latency_values else None,
                    "max": max(p95_values or latency_values)
                    if latency_values
                    else None,
                },
                "critical_path": True,
            }
            observations.append(
                self._observation(
                    channel="trace",
                    key=f"{group_id}:{window_start}:{query_time}",
                    start=start,
                    end=end,
                    available_at=max(
                        item["available_at_time_seconds"] for item in samples
                    ),
                    available_at_sequence=max(
                        item["available_at_sequence"] for item in samples
                    ),
                    payload=payload,
                    entities=[
                        _entity(source, "source", provenance="trace_resource"),
                        _entity(
                            destination, "destination", provenance="trace_resource"
                        ),
                    ],
                    subsystem="application",
                    correlations={"trace_group_id": group_id},
                    source_keys=[
                        f"trace:{item['capture_sequence']}:{source}:{destination}:{operation}"
                        for item in samples
                    ],
                    missingness=0.0,
                    availability_mask={
                        "operation": True,
                        "source": True,
                        "destination": True,
                        "latency": bool(latency_values),
                        "status": True,
                        "count": True,
                        "retries": True,
                        "critical_path": True,
                    },
                )
            )
        return observations

    def _adapt_config(
        self,
        window_start: float,
        query_time: float,
        query_watermark: int,
    ) -> list[dict[str, Any]]:
        """Derive configuration state/deltas from causally available frames."""
        latest_frame_by_time: dict[float, dict[str, Any]] = {}
        for frame in self._frames:
            if (
                frame["event_time_seconds"] > query_time
                or frame["available_at_time_seconds"] > query_time
                or frame["available_at_sequence"] > query_watermark
            ):
                continue
            previous = latest_frame_by_time.get(frame["event_time_seconds"])
            if (
                previous is None
                or frame["capture_sequence"] > previous["capture_sequence"]
            ):
                latest_frame_by_time[frame["event_time_seconds"]] = frame

        available: list[dict[str, Any]] = []
        previous_config: dict[str, Any] = {}
        for event_time in sorted(latest_frame_by_time):
            frame = latest_frame_by_time[event_time]
            flattened = _flatten_mapping(frame.get("config", {}))
            changed_paths = sorted(
                path
                for path in set(flattened) | set(previous_config)
                if flattened.get(path) != previous_config.get(path)
            )
            if changed_paths:
                revision_id = "config-revision-" + self._opaque_digest(
                    "config-revision",
                    {
                        "capture_sequence": frame["capture_sequence"],
                        "time": event_time,
                        "changes": {
                            path: flattened.get(path) for path in changed_paths
                        },
                    },
                    length=20,
                )
                for path in changed_paths:
                    operation = (
                        "remove"
                        if path not in flattened
                        else ("set" if path not in previous_config else "update")
                    )
                    available.append(
                        {
                            "path": path,
                            "value": deepcopy(flattened.get(path)),
                            "previous_value": deepcopy(previous_config.get(path)),
                            "scope": _config_scope(path),
                            "operation": operation,
                            "event_time_seconds": event_time,
                            "available_at_time_seconds": frame[
                                "available_at_time_seconds"
                            ],
                            "available_at_sequence": frame["available_at_sequence"],
                            "revision_id": revision_id,
                        }
                    )
            previous_config = flattened

        latest_by_path: dict[str, dict[str, Any]] = {}
        in_window: list[dict[str, Any]] = []
        for event in available:
            latest_by_path[event["path"]] = event
            if event["event_time_seconds"] >= window_start:
                in_window.append(event)

        selected: dict[tuple[str, float], dict[str, Any]] = {}
        for event in latest_by_path.values():
            selected[(event["path"], event["event_time_seconds"])] = {
                **event,
                "operation": (
                    event["operation"]
                    if event["event_time_seconds"] >= window_start
                    else "state"
                ),
            }
        for event in in_window:
            selected[(event["path"], event["event_time_seconds"])] = event

        observations = []
        for event in sorted(
            selected.values(),
            key=lambda item: (item["event_time_seconds"], item["path"]),
        ):
            scope = event["scope"]
            revision_id = event["revision_id"]
            payload = {
                "unit_type": "scoped_path_value",
                "path": event["path"],
                "value": deepcopy(event["value"]),
                "value_type": _value_type(event["value"]),
                "previous_value": deepcopy(event.get("previous_value")),
                "scope": scope,
                "operation": event["operation"],
                "change_time_seconds": event["event_time_seconds"],
            }
            observations.append(
                self._observation(
                    channel="config",
                    key=f"{revision_id}:{event['path']}:{event['event_time_seconds']}",
                    start=event["event_time_seconds"],
                    end=event["event_time_seconds"],
                    available_at=event["available_at_time_seconds"],
                    available_at_sequence=event["available_at_sequence"],
                    payload=payload,
                    entities=[
                        _entity(scope, "scope", provenance="configuration_scope")
                    ],
                    subsystem=_subsystem_for_config_path(event["path"]),
                    correlations={"config_revision": revision_id},
                    source_keys=[f"config:{revision_id}:{event['path']}"],
                    missingness=0.0,
                    availability_mask={
                        "path": True,
                        "typed_value": True,
                        "scope": True,
                        "operation": True,
                        "change_time": True,
                    },
                )
            )
        return observations

    def _observation(
        self,
        *,
        channel: str,
        key: str,
        start: float,
        end: float,
        available_at: float,
        available_at_sequence: int,
        payload: dict[str, Any],
        entities: list[dict[str, Any]],
        subsystem: str,
        correlations: dict[str, str],
        source_keys: Sequence[str],
        missingness: float,
        availability_mask: dict[str, bool],
    ) -> dict[str, Any]:
        if channel not in CHANNELS:
            raise CanonicalTelemetryError(f"unsupported canonical channel: {channel}")
        observation_id = "obs-" + _stable_hash(
            {
                "episode_id": self.episode_id,
                "channel": channel,
                "key": key,
                "start": start,
                "end": end,
                "payload": payload,
            },
            length=24,
        )
        delay = max(0.0, available_at - end)
        source_references = [
            "source-"
            + self._opaque_digest(
                "source-reference",
                {"source": source_key},
                length=20,
            )
            for source_key in source_keys
        ]
        metadata = {
            "event_start_time_seconds": start,
            "event_end_time_seconds": end,
            "ingest_time_seconds": available_at,
            "available_at_time_seconds": available_at,
            "available_at_sequence": available_at_sequence,
            "entities": entities,
            "primary_subsystem": subsystem if subsystem else "unknown",
            "primary_subsystem_provenance": (
                "authorized_inventory" if subsystem else "unavailable"
            ),
            "correlation_ids": dict(sorted(correlations.items())),
            "source_references": source_references,
            "data_quality": {
                "parse_confidence": 1.0,
                "missingness_fraction": round(float(missingness), 6),
                "delay_seconds": round(delay, 6),
                "availability_mask": dict(sorted(availability_mask.items())),
                "validation_flags": [],
            },
        }
        return {
            "observation_id": observation_id,
            "channel": channel,
            "window": {
                "start_time_seconds": start,
                "end_time_seconds": end,
                "start_inclusive": True,
                "end_inclusive": True,
            },
            "payload": payload,
            "metadata": metadata,
        }


def assert_inference_safe(value: Any) -> None:
    """Fail closed if a canonical inference payload contains training/oracle data."""
    violations: list[str] = []

    def visit(item: Any, path: str) -> None:
        if isinstance(item, Mapping):
            for raw_key, child in item.items():
                key = str(raw_key).lower()
                child_path = f"{path}.{raw_key}" if path else str(raw_key)
                if key in FORBIDDEN_INFERENCE_KEYS:
                    violations.append(child_path)
                visit(child, child_path)
            return
        if isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray)):
            for index, child in enumerate(item):
                visit(child, f"{path}[{index}]")
            return
        if isinstance(item, str):
            if _contains_fault_mechanism(item):
                violations.append(path)
            if _EXACT_SERVER_IDENTIFIER.search(item):
                violations.append(path)

    visit(value, "")
    if violations:
        rendered = ", ".join(sorted(set(violations))[:10])
        raise CanonicalTelemetryError(
            f"inference snapshot contains forbidden training/oracle data at: {rendered}"
        )


def _require_nonempty_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise CanonicalTelemetryError(f"{field} must be non-empty text")
    return value


def _validate_json_value(value: Any, field: str) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CanonicalTelemetryError(f"{field} contains a non-finite number")
        return
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise CanonicalTelemetryError(f"{field} contains a non-string key")
        for key, child in value.items():
            _validate_json_value(child, f"{field}.{key}")
        return
    if isinstance(value, list):
        for index, child in enumerate(value):
            _validate_json_value(child, f"{field}[{index}]")
        return
    raise CanonicalTelemetryError(f"{field} contains a non-JSON value")


def _validate_payload(
    channel: str,
    payload: Mapping[str, Any],
    *,
    window_start: float,
    window_end: float,
) -> None:
    if set(payload) != REQUIRED_PAYLOAD_FIELDS[channel]:
        missing = REQUIRED_PAYLOAD_FIELDS[channel] - set(payload)
        extra = set(payload) - REQUIRED_PAYLOAD_FIELDS[channel]
        raise CanonicalTelemetryError(
            f"{channel} payload fields are invalid; "
            f"missing={sorted(missing)}, extra={sorted(extra)}"
        )
    _require_nonempty_text(payload.get("unit_type"), f"{channel}.unit_type")

    if channel == "log":
        for field in ("template_id", "template", "event_type", "severity"):
            _require_nonempty_text(payload.get(field), f"log.{field}")
        count = payload.get("count")
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise CanonicalTelemetryError("log.count must be a positive integer")
        severity_histogram = payload.get("severity_histogram")
        if (
            not isinstance(severity_histogram, Mapping)
            or not severity_histogram
            or any(
                severity not in _SEVERITY_ORDER
                or isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
                for severity, value in severity_histogram.items()
            )
            or sum(severity_histogram.values()) != count
        ):
            raise CanonicalTelemetryError("log.severity_histogram is invalid")
        rarity = _number(payload.get("rarity"))
        burst = _number(payload.get("burst_rate_per_minute"))
        if not 0.0 <= rarity <= 1.0 or burst < 0.0:
            raise CanonicalTelemetryError("log rarity or burst rate is invalid")
        variables = payload.get("variable_summaries")
        time_features = payload.get("time_features")
        if not isinstance(variables, Mapping) or not isinstance(
            time_features,
            Mapping,
        ):
            raise CanonicalTelemetryError("log summaries must be objects")
        if set(time_features) != {"first_offset_seconds", "last_offset_seconds"}:
            raise CanonicalTelemetryError("log.time_features fields are invalid")
        first_offset = _number(time_features.get("first_offset_seconds"))
        last_offset = _number(time_features.get("last_offset_seconds"))
        if first_offset < 0 or first_offset > last_offset:
            raise CanonicalTelemetryError("log.time_features are invalid")
        _validate_json_value(variables, "log.variable_summaries")
        if _safe_details(variables) != variables:
            raise CanonicalTelemetryError(
                "log.variable_summaries contain unsafe simulator details"
            )
        return

    if channel == "metric":
        for field in ("metric_name", "unit", "scale"):
            _require_nonempty_text(payload.get(field), f"metric.{field}")
        sample_period = _number(payload.get("sample_period_seconds"))
        if sample_period <= 0:
            raise CanonicalTelemetryError(
                "metric.sample_period_seconds must be positive"
            )
        timestamps = payload.get("timestamps_seconds")
        values = payload.get("values")
        missing_mask = payload.get("missingness_mask")
        if (
            not isinstance(timestamps, list)
            or not isinstance(values, list)
            or not isinstance(missing_mask, list)
            or not timestamps
            or not len(timestamps) == len(values) == len(missing_mask)
            or not all(isinstance(missing, bool) for missing in missing_mask)
        ):
            raise CanonicalTelemetryError("metric series arrays are invalid")
        numeric_timestamps = [_number(timestamp) for timestamp in timestamps]
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
            raise CanonicalTelemetryError("metric timestamps are invalid")
        for index, (value, missing) in enumerate(
            zip(values, missing_mask, strict=True)
        ):
            if missing:
                if value is not None:
                    raise CanonicalTelemetryError(
                        "metric missingness mask disagrees with values"
                    )
            elif (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise CanonicalTelemetryError(
                    f"metric.values[{index}] must be a finite number"
                )
        statistics_payload = payload.get("statistics")
        expected_statistics = {
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
        if (
            not isinstance(statistics_payload, Mapping)
            or set(statistics_payload) != expected_statistics
        ):
            raise CanonicalTelemetryError("metric.statistics fields are invalid")
        for field, value in statistics_payload.items():
            _number(value)
        normalization = payload.get("normalization_reference")
        if (
            not isinstance(normalization, Mapping)
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
            raise CanonicalTelemetryError("metric.normalization_reference is invalid")
        _number(normalization.get("median"))
        if _number(normalization.get("iqr")) < 0:
            raise CanonicalTelemetryError("metric normalization IQR is invalid")
        _number(normalization.get("last"))
        sample_count = normalization.get("sample_count")
        if (
            isinstance(sample_count, bool)
            or not isinstance(sample_count, int)
            or sample_count < 1
        ):
            raise CanonicalTelemetryError(
                "metric normalization sample_count is invalid"
            )
        reference_end = normalization.get("reference_end_time_seconds")
        if normalization["method"] == "causal_pre_window_robust":
            if _number(reference_end) >= window_start:
                raise CanonicalTelemetryError(
                    "metric normalization reference is not pre-window"
                )
        elif reference_end is not None:
            raise CanonicalTelemetryError(
                "metric fallback reference_end_time_seconds must be null"
            )
        resource = payload.get("resource")
        if not isinstance(resource, Mapping) or set(resource) != {"entity_id"}:
            raise CanonicalTelemetryError("metric.resource is invalid")
        _require_nonempty_text(resource.get("entity_id"), "metric.resource.entity_id")
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
            _require_nonempty_text(payload.get(field), f"alert.{field}")
        if payload["status"] not in {"firing", "resolved"}:
            raise CanonicalTelemetryError("alert.status is invalid")
        if payload["severity"] not in _SEVERITY_ORDER:
            raise CanonicalTelemetryError("alert.severity is invalid")
        threshold = payload.get("threshold")
        details = payload.get("details")
        if threshold is not None and not isinstance(threshold, Mapping):
            raise CanonicalTelemetryError("alert.threshold must be an object or null")
        if not isinstance(details, Mapping):
            raise CanonicalTelemetryError("alert.details must be an object")
        duration = _number(payload.get("duration_seconds"))
        if duration < 0 or not math.isclose(
            duration,
            max(0.0, window_end - window_start),
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            raise CanonicalTelemetryError("alert.duration_seconds is invalid")
        _validate_json_value(threshold, "alert.threshold")
        _validate_json_value(details, "alert.details")
        if _safe_details(details) != details:
            raise CanonicalTelemetryError("alert.details contain unsafe data")
        return

    if channel == "trace":
        for field in ("operation", "source", "destination"):
            _require_nonempty_text(payload.get(field), f"trace.{field}")
        count = _number(payload.get("count"))
        retry_count = _number(payload.get("retry_count"))
        if count < 0 or retry_count < 0:
            raise CanonicalTelemetryError("trace counts must be non-negative")
        status_counts = payload.get("status_counts")
        if not isinstance(status_counts, Mapping) or set(status_counts) != {
            "ok",
            "error",
        }:
            raise CanonicalTelemetryError("trace.status_counts is invalid")
        ok_count = _number(status_counts.get("ok"))
        error_count = _number(status_counts.get("error"))
        if (
            ok_count < 0
            or error_count < 0
            or not math.isclose(
                ok_count + error_count,
                count,
                rel_tol=0.0,
                abs_tol=1e-5,
            )
        ):
            raise CanonicalTelemetryError("trace.status_counts do not match count")
        latency = payload.get("latency_ms")
        if not isinstance(latency, Mapping) or set(latency) != {
            "mean",
            "p95",
            "min",
            "max",
        }:
            raise CanonicalTelemetryError("trace.latency_ms is invalid")
        for field, value in latency.items():
            if value is not None and _number(value) < 0:
                raise CanonicalTelemetryError(f"trace.latency_ms.{field} is invalid")
        if not isinstance(payload.get("critical_path"), bool):
            raise CanonicalTelemetryError("trace.critical_path must be boolean")
        return

    if channel == "config":
        for field in ("path", "value_type", "scope", "operation"):
            _require_nonempty_text(payload.get(field), f"config.{field}")
        if payload["operation"] not in {"state", "set", "update", "remove"}:
            raise CanonicalTelemetryError("config.operation is invalid")
        if not payload["path"].startswith(
            (
                "simulation.",
                "topology.",
                "thresholds.",
                "workload.",
                "cooling_units.",
                "controls.",
            )
        ):
            raise CanonicalTelemetryError("config.path is outside the public schema")
        if _number(payload.get("change_time_seconds")) != window_start:
            raise CanonicalTelemetryError("config.change_time_seconds is invalid")
        if payload["value_type"] != _value_type(payload.get("value")):
            raise CanonicalTelemetryError("config.value_type is inconsistent")
        _validate_json_value(payload.get("value"), "config.value")
        _validate_json_value(payload.get("previous_value"), "config.previous_value")
        return

    raise CanonicalTelemetryError(f"unsupported canonical channel: {channel}")


def validate_canonical_snapshot(snapshot: Mapping[str, Any]) -> None:
    """Validate the public schema and causal availability invariant."""
    if not isinstance(snapshot, Mapping) or set(snapshot) != SNAPSHOT_FIELDS:
        raise CanonicalTelemetryError("canonical snapshot fields are invalid")
    if snapshot.get("schema_version") != SCHEMA_VERSION:
        raise CanonicalTelemetryError("unsupported canonical schema_version")
    _require_nonempty_text(snapshot.get("episode_id"), "episode_id")
    query_time = _number(snapshot.get("query_time_seconds"))
    if query_time < 0:
        raise CanonicalTelemetryError("query_time_seconds must be non-negative")
    query_watermark = snapshot.get("query_watermark_sequence")
    if (
        not isinstance(query_watermark, str)
        or re.fullmatch(r"cut-[0-9a-f]{32}", query_watermark) is None
    ):
        raise CanonicalTelemetryError(
            "query_watermark_sequence must be an opaque causal-cut token"
        )
    snapshot_window = snapshot.get("window")
    if (
        not isinstance(snapshot_window, Mapping)
        or set(snapshot_window) != WINDOW_FIELDS
    ):
        raise CanonicalTelemetryError("snapshot window must be an object")
    snapshot_start = _number(snapshot_window.get("start_time_seconds"))
    snapshot_end = _number(snapshot_window.get("end_time_seconds"))
    if (
        snapshot_start < 0
        or snapshot_start > snapshot_end
        or snapshot_end != query_time
        or snapshot_window.get("start_inclusive") is not True
        or snapshot_window.get("end_inclusive") is not True
    ):
        raise CanonicalTelemetryError("snapshot window is invalid")
    policy = snapshot.get("policy")
    if not isinstance(policy, Mapping) or set(policy) != POLICY_FIELDS:
        raise CanonicalTelemetryError("snapshot policy is invalid")
    if (
        policy.get("availability_predicate")
        != (
            "available_at_time_seconds <= query_time_seconds and "
            "available_at_sequence == query_watermark_sequence"
        )
        or policy.get("host_visibility") != "rack"
        or policy.get("timestamp_clock") != "simulation_seconds"
        or policy.get("label_plane") != "separate_training_sidecar"
        or policy.get("adapter_pipeline")
        != ["decode", "consolidate", "normalize", "window", "validate"]
        or _number(policy.get("metric_cadence_seconds")) <= 0
    ):
        raise CanonicalTelemetryError("snapshot policy values are invalid")
    observations = snapshot.get("observations")
    if not isinstance(observations, list):
        raise CanonicalTelemetryError("observations must be a list")
    requested_channels = snapshot.get("channels")
    if (
        not isinstance(requested_channels, list)
        or not all(isinstance(channel, str) for channel in requested_channels)
        or requested_channels
        != [channel for channel in CHANNELS if channel in requested_channels]
        or len(set(requested_channels)) != len(requested_channels)
    ):
        raise CanonicalTelemetryError("snapshot channels are invalid")
    channel_counts = snapshot.get("channel_counts")
    if not isinstance(channel_counts, Mapping) or set(channel_counts) != set(
        requested_channels
    ):
        raise CanonicalTelemetryError("snapshot channel_counts are invalid")
    expected_counts = {
        channel: sum(
            isinstance(observation, Mapping) and observation.get("channel") == channel
            for observation in observations
        )
        for channel in requested_channels
    }
    if (
        any(
            isinstance(count, bool) or not isinstance(count, int) or count < 0
            for count in channel_counts.values()
        )
        or dict(channel_counts) != expected_counts
    ):
        raise CanonicalTelemetryError(
            "snapshot channel_counts do not match observations"
        )
    snapshot_id = snapshot.get("snapshot_id")
    snapshot_without_id = dict(snapshot)
    snapshot_without_id.pop("snapshot_id", None)
    expected_snapshot_id = "snapshot-" + _stable_hash(
        snapshot_without_id,
        length=24,
    )
    if snapshot_id != expected_snapshot_id:
        raise CanonicalTelemetryError("snapshot_id does not match snapshot content")
    seen_ids: set[str] = set()
    for observation in observations:
        if (
            not isinstance(observation, Mapping)
            or set(observation) != OBSERVATION_FIELDS
        ):
            raise CanonicalTelemetryError("observation must be an object")
        observation_id = observation.get("observation_id")
        if not isinstance(observation_id, str) or not observation_id:
            raise CanonicalTelemetryError("observation_id must be a non-empty string")
        if observation_id in seen_ids:
            raise CanonicalTelemetryError(f"duplicate observation_id: {observation_id}")
        seen_ids.add(observation_id)
        channel = observation.get("channel")
        if not isinstance(channel, str) or channel not in requested_channels:
            raise CanonicalTelemetryError("observation has unsupported channel")
        window = observation.get("window")
        if not isinstance(window, Mapping) or set(window) != WINDOW_FIELDS:
            raise CanonicalTelemetryError("observation window must be an object")
        window_start = _number(window.get("start_time_seconds"))
        window_end = _number(window.get("end_time_seconds"))
        if (
            window_start < 0
            or window_start > window_end
            or window_end > query_time
            or window.get("start_inclusive") is not True
            or window.get("end_inclusive") is not True
        ):
            raise CanonicalTelemetryError("observation window is invalid")
        payload = observation.get("payload")
        if not isinstance(payload, Mapping):
            raise CanonicalTelemetryError("observation payload must be an object")
        _validate_payload(
            channel,
            payload,
            window_start=window_start,
            window_end=window_end,
        )
        metadata = observation.get("metadata")
        if not isinstance(metadata, Mapping) or set(metadata) != METADATA_FIELDS:
            raise CanonicalTelemetryError("observation metadata must be an object")
        event_start = _number(metadata.get("event_start_time_seconds"))
        event_end = _number(metadata.get("event_end_time_seconds"))
        available_at = _number(metadata.get("available_at_time_seconds"))
        ingest_time = _number(metadata.get("ingest_time_seconds"))
        if (
            event_start > event_end
            or event_end > query_time
            or event_start != window_start
            or event_end != window_end
        ):
            raise CanonicalTelemetryError(
                f"observation {observation_id} has an invalid event interval"
            )
        if available_at > query_time:
            raise CanonicalTelemetryError(
                f"observation {observation_id} was unavailable at query time"
            )
        if ingest_time != available_at:
            raise CanonicalTelemetryError(
                f"observation {observation_id} has inconsistent ingest time"
            )
        available_sequence = metadata.get("available_at_sequence")
        if available_sequence != query_watermark:
            raise CanonicalTelemetryError(
                f"observation {observation_id} violates the sequence watermark"
            )
        entities = metadata.get("entities")
        if not isinstance(entities, list):
            raise CanonicalTelemetryError("metadata.entities must be a list")
        for entity in entities:
            confidence = (
                entity.get("confidence") if isinstance(entity, Mapping) else None
            )
            if (
                not isinstance(entity, Mapping)
                or set(entity) != {"entity_id", "role", "confidence", "provenance"}
                or not isinstance(entity.get("entity_id"), str)
                or not entity.get("entity_id")
                or entity.get("role") not in ENTITY_ROLES
                or isinstance(confidence, bool)
                or not isinstance(confidence, (int, float))
                or not 0.0 <= float(confidence) <= 1.0
                or not isinstance(entity.get("provenance"), str)
                or not entity.get("provenance")
            ):
                raise CanonicalTelemetryError("invalid inference-visible entity role")
        primary_subsystem = metadata.get("primary_subsystem")
        subsystem_provenance = metadata.get("primary_subsystem_provenance")
        correlations = metadata.get("correlation_ids")
        source_references = metadata.get("source_references")
        if (
            not isinstance(primary_subsystem, str)
            or not primary_subsystem
            or not isinstance(subsystem_provenance, str)
            or not subsystem_provenance
            or not isinstance(correlations, Mapping)
            or not all(
                isinstance(key, str) and key and isinstance(value, str) and value
                for key, value in correlations.items()
            )
            or not isinstance(source_references, list)
            or not source_references
            or not all(
                isinstance(reference, str)
                and re.fullmatch(r"source-[0-9a-f]{20}", reference) is not None
                for reference in source_references
            )
        ):
            raise CanonicalTelemetryError("observation metadata is incomplete")
        quality = metadata.get("data_quality")
        if (
            not isinstance(quality, Mapping)
            or set(quality) != DATA_QUALITY_FIELDS
            or not isinstance(quality.get("availability_mask"), Mapping)
        ):
            raise CanonicalTelemetryError("data_quality.availability_mask is required")
        parse_confidence = quality.get("parse_confidence")
        missingness = quality.get("missingness_fraction")
        delay = quality.get("delay_seconds")
        if (
            isinstance(parse_confidence, bool)
            or not isinstance(parse_confidence, (int, float))
            or not 0.0 <= float(parse_confidence) <= 1.0
            or isinstance(missingness, bool)
            or not isinstance(missingness, (int, float))
            or not 0.0 <= float(missingness) <= 1.0
            or isinstance(delay, bool)
            or not isinstance(delay, (int, float))
            or float(delay) < 0.0
            or not all(
                isinstance(key, str) and isinstance(value, bool)
                for key, value in quality["availability_mask"].items()
            )
            or not isinstance(quality.get("validation_flags"), list)
            or not all(isinstance(flag, str) for flag in quality["validation_flags"])
        ):
            raise CanonicalTelemetryError("observation data_quality is invalid")
        expected_delay = max(0.0, available_at - event_end)
        if not math.isclose(
            float(delay),
            expected_delay,
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            raise CanonicalTelemetryError(
                f"observation {observation_id} has inconsistent delay metadata"
            )
    assert_inference_safe(snapshot)


def _extract_metric_samples(simulator: Any) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []

    def append(name: str, entity_id: str, value: Any) -> None:
        if name not in METRIC_DEFINITIONS:
            raise CanonicalTelemetryError(f"metric definition missing for {name}")
        if isinstance(value, bool):
            numeric = 1.0 if value else 0.0
        else:
            numeric = float(value)
        if not math.isfinite(numeric):
            return
        samples.append(
            {"name": name, "entity_id": str(entity_id), "value": round(numeric, 6)}
        )

    append("facility.total_it_power", "datacenter", simulator.total_it_power_kw)
    append(
        "facility.total_cooling_power", "datacenter", simulator.total_cooling_power_kw
    )
    append("facility.total_power", "datacenter", simulator.facility_power_kw)
    append("facility.pue", "datacenter", simulator.pue)

    for rack in sorted(simulator.racks, key=lambda item: item.rack_id):
        failed_count = sum(server.status == "failed" for server in rack.servers)
        flapping_count = sum(
            getattr(server, "health_check_status", "healthy") == "flapping"
            for server in rack.servers
        )
        mean_temperature = (
            statistics.fmean([server.temperature_c for server in rack.servers])
            if rack.servers
            else 0.0
        )
        append("rack.inlet_temperature", rack.rack_id, rack.inlet_temperature_c)
        append("rack.outlet_temperature", rack.rack_id, rack.outlet_temperature_c)
        append("rack.power", rack.rack_id, rack.total_power_kw)
        append(
            "rack.cpu_utilization", rack.rack_id, rack.average_cpu_utilization_percent
        )
        append("rack.failed_server_count", rack.rack_id, failed_count)
        append("rack.server_temperature_mean", rack.rack_id, mean_temperature)
        append(
            "rack.reported_inlet_temperature",
            rack.rack_id,
            getattr(rack, "reported_inlet_temperature_c", rack.inlet_temperature_c),
        )
        append(
            "rack.temperature_sensor_disagreement",
            rack.rack_id,
            getattr(rack, "temperature_sensor_disagreement_c", 0.0),
        )
        append(
            "rack.temperature_sensor_unhealthy",
            rack.rack_id,
            getattr(rack, "temperature_sensor_status", "normal") != "normal",
        )
        power_budget_kw = max(float(getattr(rack, "power_budget_kw", 0.0)), 1e-9)
        append(
            "rack.power_budget_utilization",
            rack.rack_id,
            rack.total_power_kw / power_budget_kw,
        )
        append("rack.health_flapping_count", rack.rack_id, flapping_count)
        append(
            "rack.thermal_throttle_factor",
            rack.rack_id,
            getattr(rack, "thermal_throttle_factor", 1.0),
        )
        append(
            "rack.network_packet_loss",
            rack.rack_id,
            getattr(rack, "network_packet_loss_percent", 0.0),
        )
        append(
            "rack.network_retransmit_rate",
            rack.rack_id,
            getattr(rack, "network_retransmit_rate", 0.0),
        )
        append(
            "rack.network_error_rate",
            rack.rack_id,
            getattr(rack, "network_error_rate", 0.0),
        )

    for unit in sorted(simulator.cooling_units, key=lambda item: item.cooling_unit_id):
        append("cooling.capacity", unit.cooling_unit_id, unit.cooling_capacity_kw)
        append(
            "cooling.supply_air_temperature",
            unit.cooling_unit_id,
            unit.supply_air_temperature_c,
        )
        append("cooling.fan_speed", unit.cooling_unit_id, unit.fan_speed_percent)

    workload = simulator.workload
    append("workload.current_demand", "workload", workload.current_demand_per_second)
    append("workload.configured_demand", "workload", workload.request_rate_per_second)
    append("workload.queue_length", "workload", workload.queue_length)
    append(
        "workload.service_capacity",
        "workload",
        workload.service_capacity_requests_per_second,
    )
    append("workload.average_latency", "workload", workload.average_latency_ms)
    append("workload.p95_latency", "workload", workload.p95_latency_ms)
    append("workload.network_demand", "workload", workload.network_demand_mbps)
    append(
        "workload.network_congestion",
        "workload",
        workload.network_congestion_ratio,
    )
    append(
        "workload.network_packet_loss",
        "workload",
        getattr(workload, "network_packet_loss_percent", 0.0),
    )
    append(
        "workload.network_retransmit_rate",
        "workload",
        getattr(workload, "network_retransmit_rate", 0.0),
    )
    append(
        "workload.network_error_rate",
        "workload",
        getattr(workload, "network_error_rate", 0.0),
    )
    append("workload.storage_demand", "workload", workload.storage_demand_iops)
    append(
        "workload.storage_utilization",
        "workload",
        workload.storage_utilization_ratio,
    )
    append("workload.error_rate", "workload", workload.application_error_rate_percent)
    append(
        "workload.dropped_requests",
        "workload",
        workload.dropped_requests_per_second,
    )
    append("workload.gpu_utilization", "workload", workload.gpu_utilization_percent)
    append(
        "control_plane.scheduler_api_latency",
        "control-plane",
        workload.scheduler_api_latency_ms,
    )
    append(
        "control_plane.scheduler_pending_operations",
        "control-plane",
        workload.scheduler_pending_operations,
    )
    healthy_server_count = sum(
        server.status == "healthy" for server in simulator.servers
    )
    append(
        "control_plane.autoscaler_effective_server_limit",
        "autoscaler",
        getattr(workload, "autoscaler_effective_server_limit", None)
        or healthy_server_count,
    )
    append(
        "control_plane.autoscaler_target_utilization",
        "autoscaler",
        getattr(workload, "autoscaler_target_utilization_percent", 65.0),
    )
    append(
        "control_plane.autoscaler_max_capacity",
        "autoscaler",
        getattr(workload, "autoscaler_max_capacity", None) or healthy_server_count,
    )
    append(
        "control_plane.autoscaler_cooldown",
        "autoscaler",
        getattr(workload, "autoscaler_cooldown_seconds", 60),
    )
    append(
        "control_plane.placement_policy_violations",
        "scheduler",
        getattr(workload, "placement_policy_violating_racks", 0),
    )
    append(
        "workload.placement_imbalance",
        "workload",
        getattr(workload, "workload_placement_imbalance_ratio", 0.0),
    )
    append(
        "observability.metrics_last_updated",
        "monitoring-pipeline",
        getattr(
            workload,
            "metrics_last_updated_sim_time_seconds",
            simulator.sim_time_seconds,
        ),
    )
    append(
        "observability.logs_last_updated",
        "monitoring-pipeline",
        getattr(
            workload, "logs_last_updated_sim_time_seconds", simulator.sim_time_seconds
        ),
    )
    append(
        "observability.telemetry_lag",
        "monitoring-pipeline",
        getattr(workload, "telemetry_lag_seconds", 0),
    )
    append(
        "observability.metrics_missing_ratio",
        "monitoring-pipeline",
        getattr(workload, "metrics_missing_ratio", 0.0),
    )
    append(
        "observability.logs_missing_ratio",
        "monitoring-pipeline",
        getattr(workload, "logs_missing_ratio", 0.0),
    )
    append(
        "application.load_balancer_backend_skew",
        "load-balancer",
        getattr(workload, "load_balancer_request_skew_ratio", 0.0),
    )
    append(
        "application.load_balancer_unhealthy_routing",
        "load-balancer",
        getattr(workload, "load_balancer_unhealthy_routing_fraction", 0.0),
    )
    append(
        "application.load_balancer_error_rate",
        "load-balancer",
        getattr(workload, "load_balancer_error_rate_percent", 0.0),
    )
    return samples


def _extract_trace_samples(simulator: Any, tick_seconds: int) -> list[dict[str, Any]]:
    workload = simulator.workload
    if not workload.running:
        return []
    states: list[Any] = (
        [tenant for _, tenant in sorted(simulator.tenants.items()) if tenant.running]
        if simulator.tenants
        else [workload]
    )
    traces = []
    for state in states:
        tenant_id = str(getattr(state, "tenant_id", workload.tenant_id))
        demand = max(0.0, float(state.current_demand_per_second))
        dropped = max(0.0, float(state.dropped_requests_per_second))
        count = demand * tick_seconds
        error_count = min(count, dropped * tick_seconds)
        retry_count = min(error_count, count) * 0.15
        workload_class = str(
            getattr(state, "active_workload_class", workload.active_workload_class)
        )
        operation = _operation_for_workload_class(workload_class)
        traces.append(
            {
                "operation": operation,
                "source_entity_id": f"tenant:{tenant_id}",
                "destination_entity_id": f"service:{workload_class}",
                "count": round(count, 6),
                "error_count": round(error_count, 6),
                "retry_count": round(retry_count, 6),
                "latency_ms": round(float(state.average_latency_ms), 6)
                if count
                else None,
                "p95_latency_ms": round(float(state.p95_latency_ms), 6)
                if count
                else None,
            }
        )
    return traces


def _extract_safe_alerts(simulator: Any) -> list[dict[str, Any]]:
    alerts = []
    host_alert_types = {
        "server_failed": (
            "HostHealthCheckFailed",
            "Host health check failures detected",
        ),
        "host_health_flapping": (
            "HostHealthFlapping",
            "Host health checks are flapping",
        ),
        "host_temperature_sensor_health": (
            "HostTemperatureSensorHealth",
            "Host temperature sensor readings are inconsistent",
        ),
        "host_capacity_throttled": (
            "HostCapacityThrottled",
            "Host service capacity is reduced",
        ),
    }
    for raw in simulator.alerts():
        if not isinstance(raw, Mapping):
            continue
        if raw.get("alert_type") in HIDDEN_ALERT_TYPES:
            continue
        alert = {
            "alert_type": str(raw.get("alert_type", "unknown")),
            "severity": _normalize_severity(raw.get("severity")),
            "target": _rack_visible_target(raw.get("target"), raw.get("details")),
            "message": str(raw.get("message", "")),
            "details": _safe_details(raw.get("details", {})),
        }
        if alert["alert_type"] in host_alert_types:
            alert["alert_type"], alert["message"] = host_alert_types[
                alert["alert_type"]
            ]
        if alert["alert_type"] == "application_errors":
            alert["alert_type"] = "ApplicationErrorRateElevated"
            alert["message"] = "Application-level error rate is elevated"
        alerts.append(alert)
    return alerts


def _extract_safe_configuration(simulator: Any) -> dict[str, Any]:
    config = simulator.config
    workload = simulator.workload
    return {
        "simulation": {
            "tick_seconds": config.simulation.tick_seconds,
            "auto_advance": config.simulation.auto_advance,
        },
        "topology": config.topology.model_dump(mode="json"),
        "thresholds": config.thresholds.model_dump(mode="json"),
        "workload": {
            "running": workload.running,
            "tenant_id": workload.tenant_id,
            "configured_request_rate_per_second": workload.request_rate_per_second,
            "workload_class": workload.workload_class,
            "active_workload_class": workload.active_workload_class,
            "workload_profile_type": workload.workload_profile_type,
            "current_profile_type": workload.current_profile_type,
            "placement_strategy": (
                "single_rack"
                if workload.placement_strategy == "rack_hotspot"
                else workload.placement_strategy
            ),
            "target_rack_id": workload.target_rack_id,
            "throttle_rate_per_second": workload.throttle_rate_per_second,
        },
        "cooling_units": {
            unit.cooling_unit_id: {
                "supply_air_temperature_c": unit.supply_air_temperature_c,
                "fan_speed_percent": unit.fan_speed_percent,
            }
            for unit in sorted(
                simulator.cooling_units, key=lambda item: item.cooling_unit_id
            )
        },
        "controls": {
            control.action_id: {
                "action_type": control.action_type,
                "status": control.status,
                "sim_time_seconds": control.sim_time_seconds,
                "details": _safe_details(control.details),
            }
            for control in simulator.controls
        },
    }


def _safe_details(value: Any) -> Any:
    if isinstance(value, Mapping):
        safe: dict[str, Any] = {}
        for raw_key, child in value.items():
            key = str(raw_key)
            lowered = key.lower()
            if lowered in _UNSAFE_DETAIL_EXACT_KEYS or any(
                part in lowered for part in _UNSAFE_DETAIL_KEY_PARTS
            ):
                continue
            safe_child = _safe_details(child)
            if _contains_fault_mechanism(safe_child):
                continue
            output_key = key.replace("application_error", "error").replace(
                "power_overloaded", "power_limit_exceeded"
            )
            safe[output_key] = safe_child
        return safe
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [
            safe_child
            for child in value
            if not _contains_fault_mechanism(safe_child := _safe_details(child))
        ]
    if isinstance(value, str):
        if _contains_fault_mechanism(value):
            return "unknown"
        return _EXACT_SERVER_IDENTIFIER.sub(
            lambda match: (
                f"rack-{match.group(1).lower()}-"
                f"{match.group(2).lower()}-{match.group(3)}"
            ),
            value,
        )
    return deepcopy(value)


def _normalize_channels(channels: Iterable[str] | None) -> tuple[str, ...]:
    if channels is None:
        return CHANNELS
    requested = {str(channel) for channel in channels}
    unsupported = requested - set(CHANNELS)
    if unsupported:
        raise CanonicalTelemetryError(
            f"unsupported channel(s): {', '.join(sorted(unsupported))}"
        )
    return tuple(channel for channel in CHANNELS if channel in requested)


def _number(value: Any, default: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        if default is not None:
            return default
        raise CanonicalTelemetryError(f"expected finite numeric value, got {value!r}")
    number = float(value)
    if not math.isfinite(number):
        if default is not None:
            return default
        raise CanonicalTelemetryError("numeric value must be finite")
    return number


def _stable_hash(value: Any, *, length: int) -> str:
    serialized = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()[:length]


def _entity(
    entity_id: str,
    role: str,
    *,
    provenance: str,
    confidence: float = 1.0,
) -> dict[str, Any]:
    if role not in ENTITY_ROLES:
        raise CanonicalTelemetryError(f"invalid entity role: {role}")
    return {
        "entity_id": str(entity_id) if entity_id else "unknown",
        "role": role,
        "confidence": max(0.0, min(1.0, float(confidence))),
        "provenance": provenance,
    }


def _event_severity(event_type: str) -> str:
    if "violation_started" in event_type:
        return "critical"
    if "resolved" in event_type:
        return "info"
    if event_type == "sim_step":
        return "debug"
    return "info"


def _normalize_severity(value: Any) -> str:
    severity = str(value or "info").lower()
    aliases = {
        "warn": "warning",
        "error": "critical",
        "fatal": "critical",
        "normal": "info",
    }
    severity = aliases.get(severity, severity)
    return severity if severity in _SEVERITY_ORDER else "info"


def _normalize_template(message: str) -> str:
    normalized = re.sub(r"\b\d+(?:\.\d+)?\b", "<number>", str(message))
    normalized = re.sub(
        r"\b(?:server|rack|cooling-unit)-[A-Za-z0-9_-]+\b",
        "<resource>",
        normalized,
    )
    return normalized


def _log_template_id(event: Mapping[str, Any]) -> str:
    return "log-template-" + _stable_hash(
        {
            "event_type": event["event_type"],
            "template": _normalize_template(str(event["message"])),
        },
        length=16,
    )


def _subsystem_for_log(event: Mapping[str, Any]) -> str:
    event_type = str(event.get("event_type", ""))
    if "sla" in event_type:
        return "application"
    if "workload" in event_type or "tenant" in event_type:
        return "workload"
    if "control" in event_type:
        return "operations"
    return "simulator"


def _subsystem_for_alert(alert: Mapping[str, Any]) -> str:
    alert_type = str(alert.get("alert_type", "")).lower()
    if "scheduler" in alert_type or "controlplane" in alert_type:
        return "control_plane"
    if "cooling" in alert_type:
        return "cooling"
    if "storage" in alert_type:
        return "storage"
    if "trafficdrop" in alert_type:
        return "network"
    if "capacitydrop" in alert_type:
        return "workload"
    if "latencystep" in alert_type:
        return "application"
    if "thermal" in alert_type:
        return "thermal"
    if "power" in alert_type:
        return "power"
    if "host" in alert_type or "server" in alert_type:
        return "hardware"
    if "queue" in alert_type:
        return "workload"
    if "error" in alert_type:
        return "application"
    return "operations"


def _subsystem_for_config_path(path: str) -> str:
    if path.startswith("cooling_units"):
        return "cooling"
    if path.startswith("thresholds.rack_"):
        return "thermal"
    if path.startswith("workload"):
        return "workload"
    if path.startswith("topology"):
        return "inventory"
    if path.startswith("controls"):
        return "operations"
    return "configuration"


def _alert_fingerprint(alert: Mapping[str, Any]) -> str:
    return "alert-" + _stable_hash(
        {
            "type": alert.get("alert_type"),
            "target": alert.get("target"),
        },
        length=20,
    )


def _alert_threshold(details: Mapping[str, Any]) -> dict[str, Any] | None:
    values = {
        key: value
        for key, value in details.items()
        if "threshold" in str(key).lower() or "limit" in str(key).lower()
    }
    return dict(sorted(values.items())) or None


def _rack_visible_target(target: Any, details: Any) -> str:
    target_text = str(target or "unknown")
    if target_text.startswith("server-"):
        if isinstance(details, Mapping) and isinstance(details.get("rack_id"), str):
            return str(details["rack_id"])
        match = re.match(r"server-(r\d+)-(row\d+)-rack(\d+)-", target_text)
        if match:
            return f"rack-{match.group(1)}-{match.group(2)}-{match.group(3)}"
        return "rack"
    return target_text


def _operation_for_workload_class(workload_class: str) -> str:
    normalized = workload_class.lower()
    if "batch" in normalized:
        return "execute_batch_task"
    if "ml" in normalized or "gpu" in normalized:
        return "run_inference"
    if "database" in normalized or "storage" in normalized:
        return "execute_query"
    return "serve_request"


def _variable_summaries(details: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    values_by_key: dict[str, list[Any]] = defaultdict(list)
    for item in details:
        for key, value in _flatten_mapping(item).items():
            values_by_key[key].append(value)
    summaries: dict[str, Any] = {}
    for key, values in sorted(values_by_key.items()):
        numeric = [
            float(value)
            for value in values
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        ]
        if len(numeric) == len(values) and numeric:
            summaries[key] = {
                "count": len(numeric),
                "min": round(min(numeric), 6),
                "max": round(max(numeric), 6),
                "mean": round(statistics.fmean(numeric), 6),
            }
        else:
            rendered = Counter(str(value) for value in values)
            summaries[key] = {
                "count": len(values),
                "top_values": [
                    {"value": value, "count": count}
                    for value, count in rendered.most_common(5)
                ],
            }
    return summaries


def _numeric_statistics(
    values: Sequence[float],
    timestamps: Sequence[float],
    values_with_missing: Sequence[float | None],
) -> dict[str, float]:
    mean = statistics.fmean(values)
    stddev = statistics.pstdev(values) if len(values) > 1 else 0.0
    observed = [
        (timestamp, value)
        for timestamp, value in zip(timestamps, values_with_missing, strict=True)
        if value is not None
    ]
    slope = 0.0
    if len(observed) > 1 and observed[-1][0] != observed[0][0]:
        slope = (observed[-1][1] - observed[0][1]) / (observed[-1][0] - observed[0][0])
    return {
        "count": float(len(values)),
        "min": round(min(values), 6),
        "max": round(max(values), 6),
        "mean": round(mean, 6),
        "median": round(statistics.median(values), 6),
        "p95": round(_percentile(values, 0.95), 6),
        "stddev": round(stddev, 6),
        "slope_per_second": round(slope, 6),
        "last": round(observed[-1][1], 6),
    }


def _percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _expected_timestamps(
    window_start: float,
    query_time: float,
    tick_seconds: int,
    *,
    observed_times: Mapping[float, Any],
) -> list[float]:
    if not observed_times:
        return []
    first_observed = min(observed_times)
    start = max(window_start, first_observed)
    timestamps = []
    current = start
    while current <= query_time + 1e-9:
        timestamps.append(round(current, 9))
        current += tick_seconds
    for observed in observed_times:
        if observed not in timestamps:
            timestamps.append(observed)
    return sorted(set(timestamps))


def _flatten_mapping(value: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for raw_key, child in sorted(value.items(), key=lambda item: str(item[0])):
        key = str(raw_key)
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(child, Mapping):
            flattened.update(_flatten_mapping(child, path))
        else:
            flattened[path] = deepcopy(child)
    return flattened


def _config_scope(path: str) -> str:
    parts = path.split(".")
    if parts[0] == "cooling_units" and len(parts) > 1:
        return parts[1]
    if parts[0] == "controls" and len(parts) > 1:
        return parts[1]
    if parts[0] == "workload":
        return "workload"
    return "datacenter"


def _value_type(value: Any) -> str:
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
    if isinstance(value, Mapping):
        return "object"
    return type(value).__name__


def _contains_fault_mechanism(value: Any) -> bool:
    if isinstance(value, str):
        lowered = value.lower()
        return any(mechanism in lowered for mechanism in FAULT_MECHANISMS)
    if isinstance(value, Mapping):
        return any(
            _contains_fault_mechanism(key) or _contains_fault_mechanism(child)
            for key, child in value.items()
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return any(_contains_fault_mechanism(child) for child in value)
    return False
