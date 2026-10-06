"""Fault request validation helpers."""

from __future__ import annotations

from pydantic import BaseModel, Field

SUPPORTED_FAULTS = {
    "cooling_degradation",
    "network_partition",
    "network_congestion_burst",
    "tor_packet_loss",
    "storage_io_saturation",
    "control_plane_degradation",
    "autoscaler_misconfiguration",
    "monitoring_pipeline_failure",
    "placement_policy_misconfiguration",
    "load_balancer_misconfiguration",
    "application_error",
    "rack_hotspot",
    "power_overload",
    "server_failure",
    "thermal_sensor_miscalibration",
    "power_budget_violation",
    "intermittent_server_failure",
    "thermal_throttling",
}

FAULT_TARGET_SCOPES = {
    "cooling_degradation": "cooling_unit",
    "rack_hotspot": "rack",
    "power_overload": "rack",
    "server_failure": "server",
    "network_partition": "rack",
    "network_congestion_burst": "workload_tenant_or_rack_path",
    "tor_packet_loss": "rack",
    "storage_io_saturation": "storage",
    "control_plane_degradation": "control_plane",
    "autoscaler_misconfiguration": "control_plane",
    "monitoring_pipeline_failure": "monitoring_pipeline",
    "placement_policy_misconfiguration": "scheduler_or_rack",
    "load_balancer_misconfiguration": "application_load_balancer",
    "application_error": "application_or_tenant",
    "thermal_sensor_miscalibration": "rack_or_server_sensor",
    "power_budget_violation": "rack",
    "intermittent_server_failure": "server",
    "thermal_throttling": "rack_or_server",
}

GLOBAL_TARGETS = {"global", "datacenter", "*"}
STORAGE_TARGETS = GLOBAL_TARGETS | {"storage", "storage-subsystem"}
CONTROL_PLANE_TARGETS = GLOBAL_TARGETS | {"control-plane", "scheduler", "api-server"}
MONITORING_TARGETS = GLOBAL_TARGETS | {"monitoring", "monitoring-pipeline", "telemetry-pipeline", "metrics-pipeline"}
APPLICATION_TARGETS = GLOBAL_TARGETS | {"application", "workload"}


class FaultRequest(BaseModel):
    fault_type: str
    target: str
    severity: float = Field(ge=0.0, le=1.0)
    duration_seconds: int = Field(default=300, gt=0)
    period_seconds: int | None = Field(default=None, gt=0)
    duty_cycle: float | None = Field(default=None, ge=0.0, le=1.0)
    failure_window_seconds: int | None = Field(default=None, gt=0)
