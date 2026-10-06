"""Configuration loading for the data center twin."""

from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path
from typing import Any

try:
    import yaml
except ModuleNotFoundError:
    yaml = None
from pydantic import BaseModel, ConfigDict, Field, model_validator
import json


class StrictConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SimulationConfig(StrictConfigModel):
    tick_seconds: int = Field(default=1, ge=1)
    initial_seed: int = 42
    auto_advance: bool = False


class TopologyConfig(StrictConfigModel):
    rooms: int = Field(default=1, ge=1)
    rows_per_room: int = Field(default=2, ge=1)
    racks_per_row: int = Field(default=4, ge=1)
    servers_per_rack: int = Field(default=10, ge=1)


class ServerConfig(StrictConfigModel):
    idle_power_kw: float = Field(default=0.08, ge=0.0)
    dynamic_power_kw: float = Field(default=0.22, ge=0.0)


class RackConfig(StrictConfigModel):
    power_limit_kw: float = Field(default=8.0, gt=0.0)
    power_budget_kw: float = Field(default=4.0, gt=0.0)
    heat_gain_factor: float = Field(default=0.15, ge=0.0)
    cooling_effect_factor: float = Field(default=0.005, ge=0.0)


class CoolingConfig(StrictConfigModel):
    units: int = Field(default=2, ge=1)
    baseline_capacity_kw: float = Field(default=50.0, gt=0.0)
    cooling_efficiency: float = Field(default=3.0, gt=0.0)
    supply_air_temperature_c: float = 20.0
    fan_speed_percent: float = Field(default=60.0, ge=0.0, le=100.0)


class ThresholdConfig(StrictConfigModel):
    rack_warning_temp_c: float = 27.0
    rack_critical_temp_c: float = 32.0
    failed_server_ratio_sla_threshold: float = Field(default=0.05, ge=0.0, le=1.0)
    workload_queue_sla_threshold: int = Field(default=1000, ge=0)

    @model_validator(mode="after")
    def validate_temperature_thresholds(self) -> "ThresholdConfig":
        if self.rack_warning_temp_c >= self.rack_critical_temp_c:
            raise ValueError("rack_warning_temp_c must be lower than rack_critical_temp_c")
        return self


class WorkloadConfig(StrictConfigModel):
    job_duration_seconds: int | None = Field(default=None, ge=0)
    tenant_id: str = "default"
    request_rate_per_second: float = Field(default=100.0, ge=0.0)
    workload_class: str = "web_service"
    workload_profile_type: str = "steady"
    workload_profile_parameters: dict[str, Any] = Field(default_factory=dict)
    trace_replay: dict[str, Any] = Field(default_factory=dict)
    cpu_cost_per_request: float = Field(default=0.001, ge=0.0)
    memory_cost_per_request_mb: float = Field(default=1.0, ge=0.0)
    network_kb_per_request: float = Field(default=8.0, ge=0.0)
    network_capacity_mbps: float = Field(default=1000.0, gt=0.0)
    storage_io_per_request: float = Field(default=1.0, ge=0.0)
    storage_capacity_iops: float = Field(default=50000.0, gt=0.0)
    base_latency_ms: float = Field(default=5.0, ge=0.0)
    service_latency_ms: float = Field(default=2.0, ge=0.0)
    network_base_latency_ms: float = Field(default=1.0, ge=0.0)
    network_congestion_penalty_ms: float = Field(default=25.0, ge=0.0)
    storage_base_latency_ms: float = Field(default=0.5, ge=0.0)
    storage_congestion_penalty_ms: float = Field(default=20.0, ge=0.0)
    latency_fault_penalty_ms: float = Field(default=10.0, ge=0.0)
    placement_strategy: str = "spread"
    noise_enabled: bool = True
    noise_stddev: float = Field(default=0.05, ge=0.0)
    target_rack_id: str | None = None
    tenants: list[dict[str, Any]] = Field(default_factory=list)


class DataCenterTwinConfig(StrictConfigModel):
    simulation: SimulationConfig = Field(default_factory=SimulationConfig)
    topology: TopologyConfig = Field(default_factory=TopologyConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    rack: RackConfig = Field(default_factory=RackConfig)
    cooling: CoolingConfig = Field(default_factory=CoolingConfig)
    thresholds: ThresholdConfig = Field(default_factory=ThresholdConfig)
    workload: WorkloadConfig = Field(default_factory=WorkloadConfig)
    ambient_temperature_c: float = Field(default=22.0, ge=-50.0, le=80.0)

    def merged(self, override: dict[str, Any] | None) -> "DataCenterTwinConfig":
        data = self.model_dump()
        if override:
            _deep_update(data, override)
        return DataCenterTwinConfig.model_validate(data)


def _deep_update(target: dict[str, Any], override: dict[str, Any]) -> None:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = value


def default_config() -> DataCenterTwinConfig:
    return DataCenterTwinConfig()


def load_config(path: str | os.PathLike[str] | None = None) -> DataCenterTwinConfig:
    raw_path = path or os.environ.get("DC_TWIN_CONFIG")
    print(f"Loading configuration from: {raw_path or 'default settings'}")
    if not raw_path:
        return default_config()
    config_path = Path(raw_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Data Center Twin configuration file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as file:
        raw_text = file.read()
    if yaml is not None:
        raw = yaml.safe_load(raw_text) or {}
    else:
        raw = json.loads(raw_text) if raw_text.strip() else {}
    base = deepcopy(default_config().model_dump())
    _deep_update(base, raw)
    return DataCenterTwinConfig.model_validate(base)
