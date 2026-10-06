"""Control action request validation helpers."""

from __future__ import annotations

from pydantic import BaseModel, Field

SUPPORTED_ACTIONS = {
    "calibrate_sensor",
    "set_cooling",
    "migrate_workload",
    "throttle_workload",
    "update_autoscaler_policy",
    "repair_monitoring_pipeline",
    "update_placement_policy",
    "update_load_balancer_config",
    "set_server_maintenance",
    "clear_server_maintenance",
}


class ControlRequest(BaseModel):
    action_type: str
    target: str | None = None
    fan_speed_percent: float | None = Field(default=None, ge=0.0, le=100.0)
    supply_air_temperature_c: float | None = None
    source_rack_id: str | None = None
    target_rack_id: str | None = None
    tenant_id: str | None = None
    workload_fraction: float | None = Field(default=None, ge=0.0, le=1.0)
    request_rate_per_second: float | None = Field(default=None, ge=0.0)
    server_id: str | None = None
    rack_id: str | None = None
    calibration_offset_c: float | None = None
    mark_untrusted: bool | None = None
    min_capacity: int | None = Field(default=None, ge=0)
    max_capacity: int | None = Field(default=None, ge=1)
    target_utilization_percent: float | None = Field(default=None, ge=1.0, le=100.0)
    cooldown_seconds: int | None = Field(default=None, ge=0)
    placement_strategy: str | None = None
    forbidden_rack_ids: list[str] | None = None
    max_server_count: int | None = Field(default=None, ge=1)
    backend_weights: dict[str, float] | None = None
    remove_backend_ids: list[str] | None = None
    add_backend_ids: list[str] | None = None
    routing_policy: str | None = None
    reset_to_equal_weights: bool | None = None
