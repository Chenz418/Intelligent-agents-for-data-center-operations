"""Workload demand generators for the simulator."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Protocol

SUPPORTED_WORKLOAD_PROFILES = {"steady", "burst", "diurnal", "maintenance_window"}
WORKLOAD_PROFILE_ALIASES = {
    "maintenance": "maintenance_window",
    "maintenance-window": "maintenance_window",
    "maintenance_window_aware": "maintenance_window",
    "maintenance-window-aware": "maintenance_window",
}


@dataclass(frozen=True)
class GeneratedWorkloadDemand:
    profile_type: str
    request_rate_per_second: float
    affected_server_ids: set[str] = field(default_factory=set)
    affected_server_workload_fraction: float = 1.0
    maintenance_active: bool = False


class WorkloadProfile(Protocol):
    profile_type: str

    def demand_at(self, sim_time_seconds: int) -> GeneratedWorkloadDemand:
        """Return generated demand for a simulation timestamp."""


@dataclass(frozen=True)
class SteadyWorkloadProfile:
    request_rate_per_second: float
    profile_type: str = "steady"

    def demand_at(self, sim_time_seconds: int) -> GeneratedWorkloadDemand:
        return GeneratedWorkloadDemand(
            profile_type=self.profile_type,
            request_rate_per_second=max(0.0, self.request_rate_per_second),
        )


@dataclass(frozen=True)
class BurstWorkloadProfile:
    baseline_rate_per_second: float
    burst_rate_per_second: float
    burst_start_time_seconds: int
    burst_duration_seconds: int
    repeat_interval_seconds: int | None = None
    profile_type: str = "burst"

    def demand_at(self, sim_time_seconds: int) -> GeneratedWorkloadDemand:
        in_burst = False
        if self.burst_duration_seconds > 0 and sim_time_seconds >= self.burst_start_time_seconds:
            offset = sim_time_seconds - self.burst_start_time_seconds
            if self.repeat_interval_seconds and self.repeat_interval_seconds > 0:
                in_burst = offset % self.repeat_interval_seconds < self.burst_duration_seconds
            else:
                in_burst = offset < self.burst_duration_seconds
        rate = self.burst_rate_per_second if in_burst else self.baseline_rate_per_second
        return GeneratedWorkloadDemand(profile_type=self.profile_type, request_rate_per_second=max(0.0, rate))


@dataclass(frozen=True)
class DiurnalWorkloadProfile:
    peak_rate_per_second: float
    trough_rate_per_second: float
    period_seconds: int
    profile_type: str = "diurnal"

    def demand_at(self, sim_time_seconds: int) -> GeneratedWorkloadDemand:
        period = max(1, self.period_seconds)
        low = min(self.peak_rate_per_second, self.trough_rate_per_second)
        high = max(self.peak_rate_per_second, self.trough_rate_per_second)
        phase = (sim_time_seconds % period) / period
        curve_position = 0.5 * (1.0 - math.cos(2.0 * math.pi * phase))
        rate = low + (high - low) * curve_position
        return GeneratedWorkloadDemand(profile_type=self.profile_type, request_rate_per_second=round(rate, 6))


@dataclass(frozen=True)
class MaintenanceWindowWorkloadProfile:
    baseline_rate_per_second: float
    maintenance_start_time_seconds: int
    maintenance_end_time_seconds: int
    affected_server_ids: set[str]
    affected_server_workload_fraction: float = 0.0
    profile_type: str = "maintenance_window"

    def demand_at(self, sim_time_seconds: int) -> GeneratedWorkloadDemand:
        active = self.maintenance_start_time_seconds <= sim_time_seconds < self.maintenance_end_time_seconds
        return GeneratedWorkloadDemand(
            profile_type=self.profile_type,
            request_rate_per_second=max(0.0, self.baseline_rate_per_second),
            affected_server_ids=set(self.affected_server_ids) if active else set(),
            affected_server_workload_fraction=_clamp(self.affected_server_workload_fraction, 0.0, 1.0),
            maintenance_active=active,
        )


def build_workload_profile(
    profile_type: str,
    request_rate_per_second: float,
    parameters: dict[str, Any] | None = None,
) -> WorkloadProfile:
    params = parameters or {}
    normalized_type = WORKLOAD_PROFILE_ALIASES.get(profile_type or "steady", profile_type or "steady")
    if normalized_type not in SUPPORTED_WORKLOAD_PROFILES:
        raise ValueError(f"unsupported workload_profile_type: {normalized_type}")

    if normalized_type == "steady":
        return SteadyWorkloadProfile(
            request_rate_per_second=float(params.get("request_rate_per_second", request_rate_per_second))
        )
    if normalized_type == "burst":
        return BurstWorkloadProfile(
            baseline_rate_per_second=float(params.get("baseline_rate_per_second", request_rate_per_second)),
            burst_rate_per_second=float(params.get("burst_rate_per_second", request_rate_per_second)),
            burst_start_time_seconds=int(params.get("burst_start_time_seconds", 0)),
            burst_duration_seconds=int(params.get("burst_duration_seconds", 0)),
            repeat_interval_seconds=_optional_int(params.get("repeat_interval_seconds")),
        )
    if normalized_type == "diurnal":
        return DiurnalWorkloadProfile(
            peak_rate_per_second=float(params.get("peak_rate_per_second", request_rate_per_second)),
            trough_rate_per_second=float(params.get("trough_rate_per_second", request_rate_per_second)),
            period_seconds=int(params.get("period_seconds", 86400)),
        )
    return MaintenanceWindowWorkloadProfile(
        baseline_rate_per_second=float(params.get("baseline_rate_per_second", request_rate_per_second)),
        maintenance_start_time_seconds=int(params.get("maintenance_start_time_seconds", 0)),
        maintenance_end_time_seconds=int(params.get("maintenance_end_time_seconds", 0)),
        affected_server_ids=set(params.get("affected_server_ids", [])),
        affected_server_workload_fraction=float(params.get("affected_server_workload_fraction", 0.0)),
    )


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _clamp(value: float, lower: float, upper: float) -> float:
    return min(upper, max(lower, value))
