"""Dataclasses used by the deterministic simulator."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

ServerStatus = Literal["healthy", "degraded", "failed", "maintenance"]
CoolingStatus = Literal["healthy", "degraded", "failed"]
ThermalStatus = Literal["normal", "warning", "critical"]
PowerStatus = Literal["normal", "overloaded"]
SlaStatus = Literal["normal", "violated"]


@dataclass
class Server:
    server_id: str
    rack_id: str
    status: ServerStatus = "healthy"
    cpu_utilization_percent: float = 0.0
    memory_utilization_percent: float = 0.0
    gpu_utilization_percent: float = 0.0
    power_kw: float = 0.0
    temperature_c: float = 22.0
    reported_temperature_c: float = 22.0
    temperature_sensor_status: str = "normal"
    temperature_sensor_bias_c: float = 0.0
    temperature_sensor_disagreement_c: float = 0.0
    thermal_throttle_factor: float = 1.0
    cpu_frequency_scale: float = 1.0
    health_check_status: str = "passing"
    health_status_change_count: int = 0
    last_health_status_change_sim_time_seconds: int | None = None
    workload_assigned: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Rack:
    rack_id: str
    row_id: str
    servers: list[Server] = field(default_factory=list)
    average_cpu_utilization_percent: float = 0.0
    total_power_kw: float = 0.0
    power_budget_kw: float = 4.0
    inlet_temperature_c: float = 22.0
    reported_inlet_temperature_c: float = 22.0
    outlet_temperature_c: float = 23.0
    reported_outlet_temperature_c: float = 23.0
    thermal_status: ThermalStatus = "normal"
    power_status: PowerStatus = "normal"
    power_budget_status: str = "normal"
    network_packet_loss_percent: float = 0.0
    network_retransmit_rate: float = 0.0
    network_error_rate: float = 0.0
    network_path_status: str = "normal"
    temperature_sensor_status: str = "normal"
    temperature_sensor_bias_c: float = 0.0
    temperature_sensor_disagreement_c: float = 0.0
    thermal_throttle_factor: float = 1.0

    @property
    def server_count(self) -> int:
        return len(self.servers)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["server_count"] = self.server_count
        return data


@dataclass
class Row:
    row_id: str
    room_id: str
    racks: list[Rack] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Room:
    room_id: str
    rows: list[Row] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CoolingUnit:
    cooling_unit_id: str
    status: CoolingStatus = "healthy"
    cooling_capacity_kw: float = 50.0
    baseline_capacity_kw: float = 50.0
    supply_air_temperature_c: float = 20.0
    fan_speed_percent: float = 60.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TenantWorkloadState:
    tenant_id: str
    running: bool = True
    job_duration_seconds: int | None = None
    job_started_at_sim_time_seconds: int | None = None
    job_completed: bool = False
    request_rate_per_second: float = 100.0
    throttle_rate_per_second: float | None = None
    workload_class: str = "web_service"
    workload_profile_type: str = "steady"
    workload_profile_parameters: dict[str, Any] = field(default_factory=dict)
    trace_replay: dict[str, Any] = field(default_factory=dict)
    trace_replay_enabled: bool = False
    trace_replay_progress: dict[str, Any] = field(default_factory=lambda: {"enabled": False})
    priority: int = 1
    quota_requests_per_second: float | None = None
    max_server_count: int | None = None
    forbidden_rack_ids: list[str] = field(default_factory=list)
    placement_strategy: str = "spread"
    target_rack_id: str | None = None
    current_demand_per_second: float = 0.0
    uncapped_demand_per_second: float = 0.0
    current_profile_type: str = "steady"
    active_workload_class: str = "web_service"
    class_resource_demand: dict[str, float] = field(default_factory=dict)
    cpu_demand: float = 0.45
    memory_demand: float = 0.35
    network_demand: float = 0.35
    storage_demand: float = 0.20
    gpu_demand: float = 0.0
    maintenance_window_active: bool = False
    maintenance_affected_server_ids: list[str] = field(default_factory=list)
    maintenance_affected_server_workload_fraction: float = 1.0
    allocated_server_ids: list[str] = field(default_factory=list)
    allocation_weights: dict[str, float] = field(default_factory=dict)
    desired_allocated_server_ids: list[str] = field(default_factory=list)
    desired_allocation_weights: dict[str, float] = field(default_factory=dict)
    queue_length: int = 0
    service_capacity_requests_per_second: float = 0.0
    processed_rate_per_second: float = 0.0
    average_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    queueing_latency_ms: float = 0.0
    service_time_latency_ms: float = 0.0
    network_demand_mbps: float = 0.0
    network_congestion_ratio: float = 0.0
    network_latency_penalty_ms: float = 0.0
    network_packet_loss_percent: float = 0.0
    network_retransmit_rate: float = 0.0
    network_error_rate: float = 0.0
    affected_rack_id: str | None = None
    storage_demand_iops: float = 0.0
    storage_utilization_ratio: float = 0.0
    storage_latency_penalty_ms: float = 0.0
    application_error_rate_percent: float = 0.0
    dropped_requests_per_second: float = 0.0
    cpu_usage_percent: float = 0.0
    memory_usage_percent: float = 0.0
    gpu_utilization_percent: float = 0.0
    sla_violation_count: int = 0
    sla_status: SlaStatus = "normal"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class WorkloadState:
    running: bool = False
    job_duration_seconds: int | None = None
    job_started_at_sim_time_seconds: int | None = None
    job_completed: bool = False
    tenant_id: str = "default"
    request_rate_per_second: float = 100.0
    throttle_rate_per_second: float | None = None
    workload_class: str = "web_service"
    workload_profile_type: str = "steady"
    workload_profile_parameters: dict[str, Any] = field(default_factory=dict)
    trace_replay: dict[str, Any] = field(default_factory=dict)
    trace_replay_enabled: bool = False
    trace_replay_progress: dict[str, Any] = field(default_factory=lambda: {"enabled": False})
    cpu_cost_per_request: float = 0.001
    memory_cost_per_request_mb: float = 1.0
    network_kb_per_request: float = 8.0
    network_capacity_mbps: float = 1000.0
    storage_io_per_request: float = 1.0
    storage_capacity_iops: float = 50000.0
    base_latency_ms: float = 5.0
    service_latency_ms: float = 2.0
    network_base_latency_ms: float = 1.0
    network_congestion_penalty_ms: float = 25.0
    storage_base_latency_ms: float = 0.5
    storage_congestion_penalty_ms: float = 20.0
    latency_fault_penalty_ms: float = 10.0
    placement_strategy: str = "spread"
    target_rack_id: str | None = None
    forbidden_rack_ids: list[str] = field(default_factory=list)
    max_server_count: int | None = None
    placement_policy_status: str = "normal"
    placement_policy_target_rack_id: str | None = None
    placement_policy_violating_racks: int = 0
    workload_placement_imbalance_ratio: float = 0.0
    autoscaler_enabled: bool = True
    autoscaler_min_capacity: int = 1
    autoscaler_max_capacity: int | None = None
    autoscaler_target_utilization_percent: float = 65.0
    autoscaler_cooldown_seconds: int = 60
    autoscaler_current_capacity_units: int | None = None
    autoscaler_effective_server_limit: int | None = None
    autoscaler_last_scale_action_time: int | None = None
    autoscaler_status: str = "normal"
    metrics_last_updated_sim_time_seconds: int = 0
    logs_last_updated_sim_time_seconds: int = 0
    telemetry_lag_seconds: int = 0
    metrics_missing_ratio: float = 0.0
    logs_missing_ratio: float = 0.0
    telemetry_pipeline_status: str = "normal"
    load_balancer_enabled: bool = True
    load_balancer_backend_server_ids: list[str] = field(default_factory=list)
    load_balancer_backend_weights: dict[str, float] = field(default_factory=dict)
    load_balancer_unhealthy_backend_ids: list[str] = field(default_factory=list)
    load_balancer_routing_policy: str = "round_robin"
    load_balancer_request_skew_ratio: float = 0.0
    load_balancer_unhealthy_routing_fraction: float = 0.0
    load_balancer_error_rate_percent: float = 0.0
    allocated_server_ids: list[str] = field(default_factory=list)
    allocation_weights: dict[str, float] = field(default_factory=dict)
    desired_allocated_server_ids: list[str] = field(default_factory=list)
    desired_allocation_weights: dict[str, float] = field(default_factory=dict)
    noise_enabled: bool = True
    noise_stddev: float = 0.05
    current_demand_per_second: float = 0.0
    uncapped_demand_per_second: float = 0.0
    current_profile_type: str = "steady"
    active_workload_class: str = "web_service"
    class_resource_demand: dict[str, float] = field(default_factory=dict)
    cpu_demand: float = 0.45
    memory_demand: float = 0.35
    network_demand: float = 0.35
    storage_demand: float = 0.20
    gpu_demand: float = 0.0
    maintenance_window_active: bool = False
    maintenance_affected_server_ids: list[str] = field(default_factory=list)
    maintenance_affected_server_workload_fraction: float = 1.0
    queue_length: int = 0
    service_capacity_requests_per_second: float = 0.0
    network_demand_mbps: float = 0.0
    network_congestion_ratio: float = 0.0
    network_latency_penalty_ms: float = 0.0
    network_packet_loss_percent: float = 0.0
    network_retransmit_rate: float = 0.0
    network_error_rate: float = 0.0
    affected_rack_id: str | None = None
    storage_demand_iops: float = 0.0
    storage_utilization_ratio: float = 0.0
    storage_latency_penalty_ms: float = 0.0
    application_error_rate_percent: float = 0.0
    dropped_requests_per_second: float = 0.0
    gpu_utilization_percent: float = 0.0
    queueing_latency_ms: float = 0.0
    service_time_latency_ms: float = 0.0
    fault_latency_penalty_ms: float = 0.0
    scheduler_api_latency_ms: float = 2.0
    scheduler_pending_operations: int = 0
    average_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    tenants: dict[str, dict[str, Any]] = field(default_factory=dict)
    active_tenant_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Fault:
    fault_id: str
    fault_type: str
    target: str
    severity: float
    duration_seconds: int
    started_at_sim_time_seconds: int
    parameters: dict[str, Any] = field(default_factory=dict)
    status: str = "active"

    def expired(self, sim_time_seconds: int) -> bool:
        return sim_time_seconds - self.started_at_sim_time_seconds >= self.duration_seconds

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ControlAction:
    action_id: str
    action_type: str
    status: str
    sim_time_seconds: int
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
