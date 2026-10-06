"""Strict metric-history models and catalog for DataCenterTwin."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, StrictFloat, StrictInt, StrictStr


class MetricPoint(BaseModel):
    """One timestamped numeric metric sample inside a series."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    sim_time_seconds: StrictInt
    value: StrictFloat


class MetricSeries(BaseModel):
    """A deterministic series for one ``(entity_id, metric_name)`` pair."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    metric_name: StrictStr
    entity_id: StrictStr
    entity_type: StrictStr
    unit: StrictStr
    metric_kind: StrictStr
    points: list[MetricPoint]


@dataclass(frozen=True)
class MetricCatalogEntry:
    metric_name: str
    entity_type: str
    unit: str
    metric_kind: str


GLOBAL_ENTITY_IDS = {
    "facility": "datacenter",
    "thermal": "datacenter",
    "power": "datacenter",
    "workload": "workload",
    "control_plane": "control-plane",
    "observability": "monitoring-pipeline",
}

GLOBAL_METRIC_CATALOG: dict[tuple[str, str], MetricCatalogEntry] = {
    ("facility", "total_it_power_kw"): MetricCatalogEntry("total_it_power_kw", "datacenter", "kilowatt", "gauge"),
    (
        "facility",
        "total_cooling_power_kw",
    ): MetricCatalogEntry("total_cooling_power_kw", "datacenter", "kilowatt", "gauge"),
    ("facility", "facility_power_kw"): MetricCatalogEntry("facility_power_kw", "datacenter", "kilowatt", "gauge"),
    ("facility", "pue"): MetricCatalogEntry("pue", "datacenter", "ratio", "gauge"),
    (
        "thermal",
        "average_rack_inlet_temperature_c",
    ): MetricCatalogEntry("average_rack_inlet_temperature_c", "datacenter", "celsius", "gauge"),
    (
        "thermal",
        "average_reported_rack_inlet_temperature_c",
    ): MetricCatalogEntry("average_reported_rack_inlet_temperature_c", "datacenter", "celsius", "gauge"),
    (
        "thermal",
        "max_rack_inlet_temperature_c",
    ): MetricCatalogEntry("max_rack_inlet_temperature_c", "datacenter", "celsius", "gauge"),
    (
        "thermal",
        "max_reported_rack_inlet_temperature_c",
    ): MetricCatalogEntry("max_reported_rack_inlet_temperature_c", "datacenter", "celsius", "gauge"),
    (
        "thermal",
        "max_rack_outlet_temperature_c",
    ): MetricCatalogEntry("max_rack_outlet_temperature_c", "datacenter", "celsius", "gauge"),
    ("thermal", "thermal_warnings"): MetricCatalogEntry("thermal_warnings", "datacenter", "count", "gauge"),
    ("thermal", "thermal_critical"): MetricCatalogEntry("thermal_critical", "datacenter", "count", "gauge"),
    (
        "thermal",
        "temperature_sensor_unhealthy_count",
    ): MetricCatalogEntry("temperature_sensor_unhealthy_count", "datacenter", "count", "gauge"),
    (
        "thermal",
        "max_temperature_sensor_disagreement_c",
    ): MetricCatalogEntry("max_temperature_sensor_disagreement_c", "datacenter", "celsius", "gauge"),
    (
        "thermal",
        "thermal_throttled_servers",
    ): MetricCatalogEntry("thermal_throttled_servers", "datacenter", "count", "gauge"),
    (
        "thermal",
        "min_thermal_throttle_factor",
    ): MetricCatalogEntry("min_thermal_throttle_factor", "datacenter", "ratio", "gauge"),
    ("power", "power_overloaded_racks"): MetricCatalogEntry(
        "power_limit_exceeded_racks", "datacenter", "count", "gauge"
    ),
    (
        "power",
        "power_budget_violating_racks",
    ): MetricCatalogEntry("power_budget_violating_racks", "datacenter", "count", "gauge"),
    (
        "power",
        "max_power_budget_utilization_ratio",
    ): MetricCatalogEntry("max_power_budget_utilization_ratio", "datacenter", "ratio", "gauge"),
    ("power", "failed_servers"): MetricCatalogEntry("failed_servers", "datacenter", "count", "gauge"),
    (
        "power",
        "average_cpu_utilization_percent",
    ): MetricCatalogEntry("average_cpu_utilization_percent", "datacenter", "percent", "gauge"),
    (
        "workload",
        "active_tenant_count",
    ): MetricCatalogEntry("workload_active_tenant_count", "workload", "count", "gauge"),
    (
        "workload",
        "current_demand_per_second",
    ): MetricCatalogEntry("workload_current_demand_per_second", "workload", "requests_per_second", "gauge"),
    (
        "workload",
        "uncapped_demand_per_second",
    ): MetricCatalogEntry("workload_uncapped_demand_per_second", "workload", "requests_per_second", "gauge"),
    (
        "workload",
        "configured_request_rate_per_second",
    ): MetricCatalogEntry("workload_configured_request_rate_per_second", "workload", "requests_per_second", "gauge"),
    ("workload", "queue_length"): MetricCatalogEntry("workload_queue_length", "workload", "requests", "gauge"),
    (
        "workload",
        "average_latency_ms",
    ): MetricCatalogEntry("workload_average_latency_ms", "workload", "milliseconds", "gauge"),
    ("workload", "p95_latency_ms"): MetricCatalogEntry("workload_p95_latency_ms", "workload", "milliseconds", "gauge"),
    (
        "workload",
        "service_capacity_requests_per_second",
    ): MetricCatalogEntry("workload_service_capacity_requests_per_second", "workload", "requests_per_second", "gauge"),
    (
        "workload",
        "network_demand_mbps",
    ): MetricCatalogEntry("workload_network_demand_mbps", "workload", "megabits_per_second", "gauge"),
    (
        "workload",
        "network_congestion_ratio",
    ): MetricCatalogEntry("workload_network_congestion_ratio", "workload", "ratio", "gauge"),
    (
        "workload",
        "network_packet_loss_percent",
    ): MetricCatalogEntry("workload_network_packet_loss_percent", "workload", "percent", "gauge"),
    (
        "workload",
        "network_retransmit_rate",
    ): MetricCatalogEntry("workload_network_retransmit_rate", "workload", "events_per_second", "gauge"),
    (
        "workload",
        "network_error_rate",
    ): MetricCatalogEntry("workload_network_error_rate", "workload", "events_per_second", "gauge"),
    (
        "workload",
        "storage_demand_iops",
    ): MetricCatalogEntry("workload_storage_demand_iops", "workload", "iops", "gauge"),
    (
        "workload",
        "storage_utilization_ratio",
    ): MetricCatalogEntry("workload_storage_utilization_ratio", "workload", "ratio", "gauge"),
    (
        "workload",
        "application_error_rate_percent",
    ): MetricCatalogEntry("workload_error_rate_percent", "workload", "percent", "gauge"),
    (
        "workload",
        "dropped_requests_per_second",
    ): MetricCatalogEntry("workload_dropped_requests_per_second", "workload", "requests_per_second", "gauge"),
    (
        "workload",
        "load_balancer_backend_skew_ratio",
    ): MetricCatalogEntry("load_balancer_backend_skew_ratio", "workload", "ratio", "gauge"),
    (
        "workload",
        "load_balancer_unhealthy_routing_fraction",
    ): MetricCatalogEntry("load_balancer_unhealthy_routing_fraction", "workload", "ratio", "gauge"),
    (
        "workload",
        "load_balancer_error_rate_percent",
    ): MetricCatalogEntry("load_balancer_error_rate_percent", "workload", "percent", "gauge"),
    (
        "workload",
        "gpu_utilization_percent",
    ): MetricCatalogEntry("workload_gpu_utilization_percent", "workload", "percent", "gauge"),
    (
        "workload",
        "allocated_server_count",
    ): MetricCatalogEntry("workload_allocated_server_count", "workload", "count", "gauge"),
    (
        "control_plane",
        "autoscaler_min_capacity",
    ): MetricCatalogEntry("autoscaler_min_capacity", "control-plane", "count", "gauge"),
    (
        "control_plane",
        "autoscaler_max_capacity",
    ): MetricCatalogEntry("autoscaler_max_capacity", "control-plane", "count", "gauge"),
    (
        "control_plane",
        "autoscaler_target_utilization_percent",
    ): MetricCatalogEntry("autoscaler_target_utilization_percent", "control-plane", "percent", "gauge"),
    (
        "control_plane",
        "autoscaler_cooldown_seconds",
    ): MetricCatalogEntry("autoscaler_cooldown_seconds", "control-plane", "seconds", "gauge"),
    (
        "control_plane",
        "autoscaler_current_capacity_units",
    ): MetricCatalogEntry("autoscaler_current_capacity_units", "control-plane", "count", "gauge"),
    (
        "control_plane",
        "autoscaler_effective_server_limit",
    ): MetricCatalogEntry("autoscaler_effective_server_limit", "control-plane", "count", "gauge"),
    (
        "control_plane",
        "placement_policy_violating_racks",
    ): MetricCatalogEntry("placement_policy_violating_racks", "control-plane", "count", "gauge"),
    (
        "control_plane",
        "workload_placement_imbalance_ratio",
    ): MetricCatalogEntry("workload_placement_imbalance_ratio", "control-plane", "ratio", "gauge"),
    (
        "observability",
        "metrics_last_updated_sim_time_seconds",
    ): MetricCatalogEntry("metrics_last_updated_sim_time_seconds", "monitoring-pipeline", "seconds", "gauge"),
    (
        "observability",
        "logs_last_updated_sim_time_seconds",
    ): MetricCatalogEntry("logs_last_updated_sim_time_seconds", "monitoring-pipeline", "seconds", "gauge"),
    (
        "observability",
        "telemetry_lag_seconds",
    ): MetricCatalogEntry("telemetry_lag_seconds", "monitoring-pipeline", "seconds", "gauge"),
    (
        "observability",
        "metrics_missing_ratio",
    ): MetricCatalogEntry("metrics_missing_ratio", "monitoring-pipeline", "ratio", "gauge"),
    (
        "observability",
        "logs_missing_ratio",
    ): MetricCatalogEntry("logs_missing_ratio", "monitoring-pipeline", "ratio", "gauge"),
}

RACK_METRIC_CATALOG: dict[str, MetricCatalogEntry] = {
    "inlet_temperature_c": MetricCatalogEntry("rack_inlet_temperature_c", "rack", "celsius", "gauge"),
    "outlet_temperature_c": MetricCatalogEntry("rack_outlet_temperature_c", "rack", "celsius", "gauge"),
    "power_kw": MetricCatalogEntry("rack_power_kw", "rack", "kilowatt", "gauge"),
    "power_budget_kw": MetricCatalogEntry("rack_power_budget_kw", "rack", "kilowatt", "gauge"),
    "network_packet_loss_percent": MetricCatalogEntry("rack_network_packet_loss_percent", "rack", "percent", "gauge"),
    "network_retransmit_rate": MetricCatalogEntry(
        "rack_network_retransmit_rate", "rack", "events_per_second", "gauge"
    ),
    "network_error_rate": MetricCatalogEntry("rack_network_error_rate", "rack", "events_per_second", "gauge"),
    "reported_inlet_temperature_c": MetricCatalogEntry(
        "rack_reported_inlet_temperature_c", "rack", "celsius", "gauge"
    ),
    "temperature_sensor_disagreement_c": MetricCatalogEntry(
        "rack_temperature_sensor_disagreement_c", "rack", "celsius", "gauge"
    ),
    "thermal_throttle_factor": MetricCatalogEntry("rack_thermal_throttle_factor", "rack", "ratio", "gauge"),
    "average_cpu_utilization_percent": MetricCatalogEntry(
        "rack_average_cpu_utilization_percent", "rack", "percent", "gauge"
    ),
}

SERVER_METRIC_CATALOG: dict[str, MetricCatalogEntry] = {
    "cpu_utilization_percent": MetricCatalogEntry("server_cpu_utilization_percent", "server", "percent", "gauge"),
    "memory_utilization_percent": MetricCatalogEntry(
        "server_memory_utilization_percent", "server", "percent", "gauge"
    ),
    "gpu_utilization_percent": MetricCatalogEntry("server_gpu_utilization_percent", "server", "percent", "gauge"),
    "power_kw": MetricCatalogEntry("server_power_kw", "server", "kilowatt", "gauge"),
    "temperature_c": MetricCatalogEntry("server_temperature_c", "server", "celsius", "gauge"),
    "reported_temperature_c": MetricCatalogEntry("server_reported_temperature_c", "server", "celsius", "gauge"),
    "temperature_sensor_disagreement_c": MetricCatalogEntry(
        "server_temperature_sensor_disagreement_c", "server", "celsius", "gauge"
    ),
    "thermal_throttle_factor": MetricCatalogEntry("server_thermal_throttle_factor", "server", "ratio", "gauge"),
    "cpu_frequency_scale": MetricCatalogEntry("server_cpu_frequency_scale", "server", "ratio", "gauge"),
    "health_status_change_count": MetricCatalogEntry(
        "server_health_status_change_count", "server", "count", "counter"
    ),
    "workload_assigned": MetricCatalogEntry("server_workload_assigned", "server", "requests_per_second", "gauge"),
}

COOLING_UNIT_METRIC_CATALOG: dict[str, MetricCatalogEntry] = {
    "cooling_capacity_kw": MetricCatalogEntry("cooling_capacity_kw", "cooling_unit", "kilowatt", "gauge"),
    "supply_air_temperature_c": MetricCatalogEntry(
        "cooling_supply_air_temperature_c", "cooling_unit", "celsius", "gauge"
    ),
    "fan_speed_percent": MetricCatalogEntry("cooling_fan_speed_percent", "cooling_unit", "percent", "gauge"),
}

TENANT_METRIC_CATALOG: dict[str, MetricCatalogEntry] = {
    "job_duration_seconds": MetricCatalogEntry("tenant_job_duration_seconds", "tenant", "seconds", "gauge"),
    "job_started_at_sim_time_seconds": MetricCatalogEntry(
        "tenant_job_started_at_sim_time_seconds", "tenant", "seconds", "gauge"
    ),
    "request_rate_per_second": MetricCatalogEntry(
        "tenant_request_rate_per_second", "tenant", "requests_per_second", "gauge"
    ),
    "throttle_rate_per_second": MetricCatalogEntry(
        "tenant_throttle_rate_per_second", "tenant", "requests_per_second", "gauge"
    ),
    "priority": MetricCatalogEntry("tenant_priority", "tenant", "count", "gauge"),
    "quota_requests_per_second": MetricCatalogEntry(
        "tenant_quota_requests_per_second", "tenant", "requests_per_second", "gauge"
    ),
    "max_server_count": MetricCatalogEntry("tenant_max_server_count", "tenant", "count", "gauge"),
    "current_demand_per_second": MetricCatalogEntry(
        "tenant_current_demand_per_second", "tenant", "requests_per_second", "gauge"
    ),
    "uncapped_demand_per_second": MetricCatalogEntry(
        "tenant_uncapped_demand_per_second", "tenant", "requests_per_second", "gauge"
    ),
    "cpu_demand": MetricCatalogEntry("tenant_cpu_demand", "tenant", "ratio", "gauge"),
    "memory_demand": MetricCatalogEntry("tenant_memory_demand", "tenant", "ratio", "gauge"),
    "network_demand": MetricCatalogEntry("tenant_network_demand", "tenant", "ratio", "gauge"),
    "storage_demand": MetricCatalogEntry("tenant_storage_demand", "tenant", "ratio", "gauge"),
    "gpu_demand": MetricCatalogEntry("tenant_gpu_demand", "tenant", "ratio", "gauge"),
    "maintenance_affected_server_workload_fraction": MetricCatalogEntry(
        "tenant_maintenance_affected_server_workload_fraction", "tenant", "ratio", "gauge"
    ),
    "queue_length": MetricCatalogEntry("tenant_queue_length", "tenant", "requests", "gauge"),
    "service_capacity_requests_per_second": MetricCatalogEntry(
        "tenant_service_capacity_requests_per_second", "tenant", "requests_per_second", "gauge"
    ),
    "processed_rate_per_second": MetricCatalogEntry(
        "tenant_processed_rate_per_second", "tenant", "requests_per_second", "gauge"
    ),
    "average_latency_ms": MetricCatalogEntry("tenant_average_latency_ms", "tenant", "milliseconds", "gauge"),
    "p95_latency_ms": MetricCatalogEntry("tenant_p95_latency_ms", "tenant", "milliseconds", "gauge"),
    "queueing_latency_ms": MetricCatalogEntry("tenant_queueing_latency_ms", "tenant", "milliseconds", "gauge"),
    "service_time_latency_ms": MetricCatalogEntry(
        "tenant_service_time_latency_ms", "tenant", "milliseconds", "gauge"
    ),
    "network_demand_mbps": MetricCatalogEntry(
        "tenant_network_demand_mbps", "tenant", "megabits_per_second", "gauge"
    ),
    "network_congestion_ratio": MetricCatalogEntry("tenant_network_congestion_ratio", "tenant", "ratio", "gauge"),
    "network_latency_penalty_ms": MetricCatalogEntry(
        "tenant_network_latency_penalty_ms", "tenant", "milliseconds", "gauge"
    ),
    "network_packet_loss_percent": MetricCatalogEntry(
        "tenant_network_packet_loss_percent", "tenant", "percent", "gauge"
    ),
    "network_retransmit_rate": MetricCatalogEntry(
        "tenant_network_retransmit_rate", "tenant", "events_per_second", "gauge"
    ),
    "network_error_rate": MetricCatalogEntry(
        "tenant_network_error_rate", "tenant", "events_per_second", "gauge"
    ),
    "storage_demand_iops": MetricCatalogEntry("tenant_storage_demand_iops", "tenant", "iops", "gauge"),
    "storage_utilization_ratio": MetricCatalogEntry(
        "tenant_storage_utilization_ratio", "tenant", "ratio", "gauge"
    ),
    "storage_latency_penalty_ms": MetricCatalogEntry(
        "tenant_storage_latency_penalty_ms", "tenant", "milliseconds", "gauge"
    ),
    "application_error_rate_percent": MetricCatalogEntry("tenant_error_rate_percent", "tenant", "percent", "gauge"),
    "dropped_requests_per_second": MetricCatalogEntry(
        "tenant_dropped_requests_per_second", "tenant", "requests_per_second", "gauge"
    ),
    "cpu_usage_percent": MetricCatalogEntry("tenant_cpu_usage_percent", "tenant", "percent", "gauge"),
    "memory_usage_percent": MetricCatalogEntry("tenant_memory_usage_percent", "tenant", "percent", "gauge"),
    "gpu_utilization_percent": MetricCatalogEntry("tenant_gpu_utilization_percent", "tenant", "percent", "gauge"),
    "sla_violation_count": MetricCatalogEntry("tenant_sla_violation_count", "tenant", "count", "counter"),
}

_ALLOCATED_SERVER_COUNT = "__allocated_server_count__"
GLOBAL_SUMMARY_FIELDS: dict[tuple[str, str], str] = {
    ("facility", "total_it_power_kw"): "total_it_power_kw",
    ("facility", "total_cooling_power_kw"): "total_cooling_power_kw",
    ("facility", "facility_power_kw"): "facility_power_kw",
    ("facility", "pue"): "pue",
    ("thermal", "average_rack_inlet_temperature_c"): "average_rack_inlet_temperature_c",
    ("thermal", "average_reported_rack_inlet_temperature_c"): "average_reported_rack_inlet_temperature_c",
    ("thermal", "max_rack_inlet_temperature_c"): "max_rack_inlet_temperature_c",
    ("thermal", "max_reported_rack_inlet_temperature_c"): "max_reported_rack_inlet_temperature_c",
    ("thermal", "max_rack_outlet_temperature_c"): "max_rack_outlet_temperature_c",
    ("thermal", "thermal_warnings"): "thermal_warnings",
    ("thermal", "thermal_critical"): "thermal_critical",
    ("thermal", "temperature_sensor_unhealthy_count"): "temperature_sensor_unhealthy_count",
    ("thermal", "max_temperature_sensor_disagreement_c"): "max_temperature_sensor_disagreement_c",
    ("thermal", "thermal_throttled_servers"): "thermal_throttled_servers",
    ("thermal", "min_thermal_throttle_factor"): "min_thermal_throttle_factor",
    ("power", "power_overloaded_racks"): "power_overloaded_racks",
    ("power", "power_budget_violating_racks"): "power_budget_violating_racks",
    ("power", "max_power_budget_utilization_ratio"): "max_power_budget_utilization_ratio",
    ("power", "failed_servers"): "failed_servers",
    ("power", "average_cpu_utilization_percent"): "average_cpu_utilization_percent",
    ("workload", "active_tenant_count"): "workload_active_tenant_count",
    ("workload", "current_demand_per_second"): "workload_current_demand_per_second",
    ("workload", "uncapped_demand_per_second"): "workload_uncapped_demand_per_second",
    ("workload", "configured_request_rate_per_second"): "workload_configured_request_rate_per_second",
    ("workload", "queue_length"): "workload_queue_length",
    ("workload", "average_latency_ms"): "workload_average_latency_ms",
    ("workload", "p95_latency_ms"): "workload_p95_latency_ms",
    ("workload", "service_capacity_requests_per_second"): "workload_service_capacity_requests_per_second",
    ("workload", "network_demand_mbps"): "workload_network_demand_mbps",
    ("workload", "network_congestion_ratio"): "workload_network_congestion_ratio",
    ("workload", "network_packet_loss_percent"): "workload_network_packet_loss_percent",
    ("workload", "network_retransmit_rate"): "workload_network_retransmit_rate",
    ("workload", "network_error_rate"): "workload_network_error_rate",
    ("workload", "storage_demand_iops"): "workload_storage_demand_iops",
    ("workload", "storage_utilization_ratio"): "workload_storage_utilization_ratio",
    ("workload", "application_error_rate_percent"): "workload_application_error_rate_percent",
    ("workload", "dropped_requests_per_second"): "workload_dropped_requests_per_second",
    ("workload", "load_balancer_backend_skew_ratio"): "load_balancer_backend_skew_ratio",
    ("workload", "load_balancer_unhealthy_routing_fraction"): "load_balancer_unhealthy_routing_fraction",
    ("workload", "load_balancer_error_rate_percent"): "load_balancer_error_rate_percent",
    ("workload", "gpu_utilization_percent"): "workload_gpu_utilization_percent",
    ("workload", "allocated_server_count"): _ALLOCATED_SERVER_COUNT,
    ("control_plane", "autoscaler_min_capacity"): "autoscaler_min_capacity",
    ("control_plane", "autoscaler_max_capacity"): "autoscaler_max_capacity",
    ("control_plane", "autoscaler_target_utilization_percent"): "autoscaler_target_utilization_percent",
    ("control_plane", "autoscaler_cooldown_seconds"): "autoscaler_cooldown_seconds",
    ("control_plane", "autoscaler_current_capacity_units"): "autoscaler_current_capacity_units",
    ("control_plane", "autoscaler_effective_server_limit"): "autoscaler_effective_server_limit",
    ("control_plane", "placement_policy_violating_racks"): "placement_policy_violating_racks",
    ("control_plane", "workload_placement_imbalance_ratio"): "workload_placement_imbalance_ratio",
    ("observability", "metrics_last_updated_sim_time_seconds"): "metrics_last_updated_sim_time_seconds",
    ("observability", "logs_last_updated_sim_time_seconds"): "logs_last_updated_sim_time_seconds",
    ("observability", "telemetry_lag_seconds"): "telemetry_lag_seconds",
    ("observability", "metrics_missing_ratio"): "metrics_missing_ratio",
    ("observability", "logs_missing_ratio"): "logs_missing_ratio",
}
RACK_FIELD_ATTRIBUTES = {field_name: field_name for field_name in RACK_METRIC_CATALOG}
RACK_FIELD_ATTRIBUTES["power_kw"] = "total_power_kw"
SERVER_FIELD_ATTRIBUTES = {field_name: field_name for field_name in SERVER_METRIC_CATALOG}
COOLING_UNIT_FIELD_ATTRIBUTES = {field_name: field_name for field_name in COOLING_UNIT_METRIC_CATALOG}


def metric_catalog_records() -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for source, catalog in (
        ("global", GLOBAL_METRIC_CATALOG.values()),
        ("rack", RACK_METRIC_CATALOG.values()),
        ("server", SERVER_METRIC_CATALOG.values()),
        ("cooling_unit", COOLING_UNIT_METRIC_CATALOG.values()),
        ("tenant", TENANT_METRIC_CATALOG.values()),
    ):
        for entry in catalog:
            records.append({"source": source, **entry.__dict__})
    return sorted(records, key=lambda item: (item["entity_type"], item["metric_name"], item["source"]))


MetricPointRecord = tuple[str, str, str, str, str, int, float]
METRIC_NAME_INDEX = 0
ENTITY_ID_INDEX = 1
ENTITY_TYPE_INDEX = 2
UNIT_INDEX = 3
METRIC_KIND_INDEX = 4
SIM_TIME_SECONDS_INDEX = 5
VALUE_INDEX = 6


def metric_points_from_structured_metrics(metrics: dict[str, Any], sim_time_seconds: int) -> list[MetricPointRecord]:
    points: list[MetricPointRecord] = []
    for (section, field_name), entry in sorted(GLOBAL_METRIC_CATALOG.items()):
        section_values = metrics.get(section)
        if isinstance(section_values, dict):
            _append_point(
                points,
                entry,
                GLOBAL_ENTITY_IDS[section],
                section_values.get(field_name),
                sim_time_seconds,
            )

    _append_entity_metrics(points, metrics.get("racks"), RACK_METRIC_CATALOG, sim_time_seconds)
    _append_entity_metrics(points, metrics.get("servers"), SERVER_METRIC_CATALOG, sim_time_seconds)
    _append_entity_metrics(points, metrics.get("cooling_units"), COOLING_UNIT_METRIC_CATALOG, sim_time_seconds)
    _append_entity_metrics(points, metrics.get("tenants"), TENANT_METRIC_CATALOG, sim_time_seconds)
    return sorted(
        points,
        key=lambda point: (
            point[ENTITY_TYPE_INDEX],
            point[ENTITY_ID_INDEX],
            point[METRIC_NAME_INDEX],
        ),
    )


def metric_points_from_runtime_state(
    summary: dict[str, Any],
    racks: list[Any],
    cooling_units: list[Any],
    tenants: dict[str, Any],
    sim_time_seconds: int,
) -> list[MetricPointRecord]:
    points: list[MetricPointRecord] = []
    for key, entry in sorted(GLOBAL_METRIC_CATALOG.items()):
        source_field = GLOBAL_SUMMARY_FIELDS[key]
        value = (
            len(summary.get("workload_allocated_server_ids") or [])
            if source_field == _ALLOCATED_SERVER_COUNT
            else summary.get(source_field)
        )
        _append_point(points, entry, GLOBAL_ENTITY_IDS[key[0]], value, sim_time_seconds)

    for rack in sorted(racks, key=lambda item: item.rack_id):
        for field_name, entry in sorted(RACK_METRIC_CATALOG.items()):
            _append_point(
                points,
                entry,
                rack.rack_id,
                getattr(rack, RACK_FIELD_ATTRIBUTES[field_name]),
                sim_time_seconds,
            )
        for server in sorted(rack.servers, key=lambda item: item.server_id):
            for field_name, entry in sorted(SERVER_METRIC_CATALOG.items()):
                _append_point(
                    points,
                    entry,
                    server.server_id,
                    getattr(server, SERVER_FIELD_ATTRIBUTES[field_name]),
                    sim_time_seconds,
                )

    for unit in sorted(cooling_units, key=lambda item: item.cooling_unit_id):
        for field_name, entry in sorted(COOLING_UNIT_METRIC_CATALOG.items()):
            _append_point(
                points,
                entry,
                unit.cooling_unit_id,
                getattr(unit, COOLING_UNIT_FIELD_ATTRIBUTES[field_name]),
                sim_time_seconds,
            )

    for tenant_id, tenant in sorted(tenants.items()):
        _append_entity_metrics(
            points,
            {tenant_id: tenant.to_dict()},
            TENANT_METRIC_CATALOG,
            sim_time_seconds,
        )
    return sorted(
        points,
        key=lambda point: (
            point[ENTITY_TYPE_INDEX],
            point[ENTITY_ID_INDEX],
            point[METRIC_NAME_INDEX],
        ),
    )


def metric_series_from_points(
    points: list[MetricPointRecord],
    start_time_seconds: int | float,
    end_time_seconds: int | float,
    entity_id: str | None = None,
    entity_type: str | None = None,
    metric_name: str | None = None,
) -> list[MetricSeries]:
    grouped: dict[tuple[str, str], list[MetricPointRecord]] = {}
    for point in points:
        point_time = point[SIM_TIME_SECONDS_INDEX]
        if point_time < start_time_seconds or point_time >= end_time_seconds:
            continue
        if entity_id is not None and point[ENTITY_ID_INDEX] != entity_id:
            continue
        if entity_type is not None and point[ENTITY_TYPE_INDEX] != entity_type:
            continue
        if metric_name is not None and point[METRIC_NAME_INDEX] != metric_name:
            continue
        grouped.setdefault((point[ENTITY_ID_INDEX], point[METRIC_NAME_INDEX]), []).append(point)

    series_records: list[MetricSeries] = []
    for (group_entity_id, group_metric_name), group_points in sorted(grouped.items()):
        ordered_points = sorted(group_points, key=lambda point: point[SIM_TIME_SECONDS_INDEX])
        first = ordered_points[0]
        strict_points = [
            MetricPoint(
                sim_time_seconds=point[SIM_TIME_SECONDS_INDEX],
                value=point[VALUE_INDEX],
            )
            for point in ordered_points
        ]
        series_records.append(
            MetricSeries(
                metric_name=group_metric_name,
                entity_id=group_entity_id,
                entity_type=first[ENTITY_TYPE_INDEX],
                unit=first[UNIT_INDEX],
                metric_kind=first[METRIC_KIND_INDEX],
                points=strict_points,
            )
        )
    return series_records


def _append_entity_metrics(
    points: list[MetricPointRecord],
    records: Any,
    catalog: dict[str, MetricCatalogEntry],
    sim_time_seconds: int,
) -> None:
    if not isinstance(records, dict):
        return
    for entity_id, values in sorted(records.items()):
        if not isinstance(entity_id, str) or not isinstance(values, dict):
            continue
        for field_name, entry in sorted(catalog.items()):
            _append_point(points, entry, entity_id, values.get(field_name), sim_time_seconds)


def _append_point(
    points: list[MetricPointRecord],
    entry: MetricCatalogEntry,
    entity_id: str,
    value: Any,
    sim_time_seconds: int,
) -> None:
    if not _is_numeric_metric_value(value):
        return
    points.append(
        (
            entry.metric_name,
            entity_id,
            entry.entity_type,
            entry.unit,
            entry.metric_kind,
            int(sim_time_seconds),
            float(value),
        )
    )


def _is_numeric_metric_value(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)
