"""Deterministic discrete-time data center twin simulator."""

from __future__ import annotations

import logging
import random
import secrets
import threading
from copy import deepcopy
from typing import Any

from dc_twin.canonical_telemetry import (
    CHANNELS as CANONICAL_TELEMETRY_CHANNELS,
    CanonicalTelemetryRecorder,
    validate_canonical_snapshot,
)
from dc_twin.config import DataCenterTwinConfig, default_config
from dc_twin.controls import ControlRequest, SUPPORTED_ACTIONS
from dc_twin.faults import (
    APPLICATION_TARGETS,
    CONTROL_PLANE_TARGETS,
    MONITORING_TARGETS,
    STORAGE_TARGETS,
    FaultRequest,
    SUPPORTED_FAULTS,
)
from dc_twin.logging_utils import log_event
from dc_twin.metric_history import (
    MetricSeries,
    MetricPointRecord,
    metric_points_from_runtime_state,
    metric_series_from_points,
)
from dc_twin.models import (
    ControlAction,
    CoolingUnit,
    Fault,
    Rack,
    Room,
    Row,
    Server,
    TenantWorkloadState,
    WorkloadState,
)
from dc_twin.workload_classes import BASELINE_WORKLOAD_CLASS, build_workload_class
from dc_twin.workload_profiles import GeneratedWorkloadDemand, WorkloadProfile, build_workload_profile
from dc_twin.workload_traces import TraceReplayEngine, TraceReplayError, TraceReplayRecord

MAX_RECORDED_EVENTS = 500
MAX_RECORDED_METRIC_POINTS = 200_000


class SimulationError(ValueError):
    """Raised for invalid simulation inputs."""


class DataCenterSimulator:
    def __init__(
        self,
        config: DataCenterTwinConfig | None = None,
        logger: logging.Logger | None = None,
        *,
        telemetry_opaque_key: bytes | str | None = None,
    ):
        self.base_config = config or default_config()
        self.logger = logger or logging.getLogger("dc-twin")
        self._telemetry_opaque_key = (
            secrets.token_bytes(32)
            if telemetry_opaque_key is None
            else telemetry_opaque_key
        )
        self.lock = threading.RLock()
        self._fault_counter = 0
        self._action_counter = 0
        self._event_counter = 0
        self._episode_counter = 0
        self._episode_id = "episode-0000"
        self._event_log: list[dict[str, Any]] = []
        self._metric_history_points: list[MetricPointRecord] = []
        self.reset(seed=self.base_config.simulation.initial_seed)

    def reset(self, seed: int | None = None, config_override: dict[str, Any] | None = None) -> dict[str, Any]:
        with self.lock:
            snapshot = self.snapshot_state() if hasattr(self, "config") else None
            try:
                next_config = self.base_config.merged(config_override)
                self._episode_counter += 1
                self._episode_id = f"episode-{self._episode_counter:04d}"
                self._event_counter = 0
                self._event_log = []
                self.config = next_config
                self.seed = self.config.simulation.initial_seed if seed is None else seed
                self.rng = random.Random(self.seed)
                self.sim_time_seconds = 0
                self.rooms: list[Room] = []
                self.cooling_units: list[CoolingUnit] = []
                self.workload = WorkloadState(**self.config.workload.model_dump())
                self.tenants: dict[str, TenantWorkloadState] = {}
                self._tenant_profiles: dict[str, WorkloadProfile] = {}
                self._tenant_classes = {}
                self._workload_trace: TraceReplayEngine | None = None
                self._tenant_traces: dict[str, TraceReplayEngine] = {}
                self._multi_tenant_enabled = False
                self._workload_class = build_workload_class(self.workload.workload_class)
                self._apply_workload_class_state()
                self._configure_workload_trace(self.workload.trace_replay)
                self._workload_profile = build_workload_profile(
                    self.workload.workload_profile_type,
                    self.workload.request_rate_per_second,
                    self.workload.workload_profile_parameters,
                )
                self.workload.workload_profile_type = self._workload_profile.profile_type
                self.workload.current_profile_type = self._workload_profile.profile_type
                self.active_faults: dict[str, Fault] = {}
                self._load_balancer_fault_baseline: dict[str, Any] | None = None
                self.controls: list[ControlAction] = []
                self.sla_status = "normal"
                self._last_sla_status = "normal"
                self._last_queue_update_sim_time = self.sim_time_seconds
                self._fault_counter = 0
                self._action_counter = 0
                self._build_topology()
                self._telemetry_recorder = CanonicalTelemetryRecorder(
                    episode_id=self._episode_id,
                    tick_seconds=self.config.simulation.tick_seconds,
                    opaque_key=self._telemetry_opaque_key,
                )
                if self.workload.tenants:
                    self._configure_tenants(self.workload.tenants)
                self._recalculate()
                self._metric_history_points = []
                self._record_metric_history_sample()
                self._record_event("sim_reset", "Simulation reset", {"seed": self.seed})
                self._capture_canonical_telemetry("sim_reset")
                return self.state_summary()
            except ValueError as error:
                if snapshot is not None:
                    self.restore_state(snapshot)
                raise SimulationError(str(error)) from error

    def snapshot_state(self) -> dict[str, Any]:
        with self.lock:
            return {
                "config": deepcopy(self.config),
                "seed": self.seed,
                "rng_state": self.rng.getstate(),
                "sim_time_seconds": self.sim_time_seconds,
                "rooms": deepcopy(self.rooms),
                "cooling_units": deepcopy(self.cooling_units),
                "workload": deepcopy(self.workload),
                "tenants": deepcopy(self.tenants),
                "tenant_profiles": deepcopy(self._tenant_profiles),
                "tenant_classes": deepcopy(self._tenant_classes),
                "workload_trace": deepcopy(self._workload_trace),
                "tenant_traces": deepcopy(self._tenant_traces),
                "multi_tenant_enabled": self._multi_tenant_enabled,
                "workload_class": deepcopy(self._workload_class),
                "workload_profile": deepcopy(self._workload_profile),
                "active_faults": deepcopy(self.active_faults),
                "load_balancer_fault_baseline": deepcopy(
                    self._load_balancer_fault_baseline
                ),
                "controls": deepcopy(self.controls),
                "sla_status": self.sla_status,
                "last_sla_status": self._last_sla_status,
                "last_queue_update_sim_time": self._last_queue_update_sim_time,
                "fault_counter": self._fault_counter,
                "action_counter": self._action_counter,
                "event_counter": self._event_counter,
                "episode_counter": self._episode_counter,
                "episode_id": self._episode_id,
                "event_log": deepcopy(self._event_log),
                "telemetry_recorder": deepcopy(self._telemetry_recorder),
                "metric_history_points": deepcopy(self._metric_history_points),
                "total_it_power_kw": self.total_it_power_kw,
                "total_cooling_power_kw": self.total_cooling_power_kw,
                "facility_power_kw": self.facility_power_kw,
                "pue": self.pue,
            }

    def restore_state(self, snapshot: dict[str, Any]) -> None:
        with self.lock:
            self.config = snapshot["config"]
            self.seed = snapshot["seed"]
            self.rng = random.Random()
            self.rng.setstate(snapshot["rng_state"])
            self.sim_time_seconds = snapshot["sim_time_seconds"]
            self.rooms = snapshot["rooms"]
            self.cooling_units = snapshot["cooling_units"]
            self.workload = snapshot["workload"]
            self.tenants = snapshot["tenants"]
            self._tenant_profiles = snapshot["tenant_profiles"]
            self._tenant_classes = snapshot["tenant_classes"]
            self._workload_trace = snapshot["workload_trace"]
            self._tenant_traces = snapshot["tenant_traces"]
            self._multi_tenant_enabled = snapshot["multi_tenant_enabled"]
            self._workload_class = snapshot["workload_class"]
            self._workload_profile = snapshot["workload_profile"]
            self.active_faults = snapshot["active_faults"]
            self._load_balancer_fault_baseline = snapshot.get(
                "load_balancer_fault_baseline"
            )
            self.controls = snapshot["controls"]
            self.sla_status = snapshot["sla_status"]
            self._last_sla_status = snapshot["last_sla_status"]
            self._last_queue_update_sim_time = snapshot["last_queue_update_sim_time"]
            self._fault_counter = snapshot["fault_counter"]
            self._action_counter = snapshot["action_counter"]
            self._event_counter = snapshot["event_counter"]
            self._episode_counter = snapshot["episode_counter"]
            self._episode_id = snapshot["episode_id"]
            self._event_log = snapshot["event_log"]
            self._telemetry_recorder = snapshot["telemetry_recorder"]
            self._metric_history_points = snapshot.get("metric_history_points", [])
            self.total_it_power_kw = snapshot["total_it_power_kw"]
            self.total_cooling_power_kw = snapshot["total_cooling_power_kw"]
            self.facility_power_kw = snapshot["facility_power_kw"]
            self.pue = snapshot["pue"]

    def _build_topology(self) -> None:
        topo = self.config.topology
        for room_index in range(1, topo.rooms + 1):
            room_id = f"room-{room_index}"
            room = Room(room_id=room_id)
            for row_index in range(1, topo.rows_per_room + 1):
                row_id = f"r{room_index}-row{row_index}"
                row = Row(row_id=row_id, room_id=room_id)
                for rack_index in range(1, topo.racks_per_row + 1):
                    rack_id = f"rack-r{room_index}-row{row_index}-{rack_index:02d}"
                    rack = Rack(
                        rack_id=rack_id,
                        row_id=row_id,
                        power_budget_kw=self.config.rack.power_budget_kw,
                    )
                    for server_index in range(1, topo.servers_per_rack + 1):
                        server_id = f"server-r{room_index}-row{row_index}-rack{rack_index:02d}-{server_index:02d}"
                        rack.servers.append(Server(server_id=server_id, rack_id=rack_id))
                    row.racks.append(rack)
                room.rows.append(row)
            self.rooms.append(room)
        for index in range(1, self.config.cooling.units + 1):
            self.cooling_units.append(
                CoolingUnit(
                    cooling_unit_id=f"cooling-unit-{index}",
                    cooling_capacity_kw=self.config.cooling.baseline_capacity_kw,
                    baseline_capacity_kw=self.config.cooling.baseline_capacity_kw,
                    supply_air_temperature_c=self.config.cooling.supply_air_temperature_c,
                    fan_speed_percent=self.config.cooling.fan_speed_percent,
                )
            )

    def step(self, ticks: int = 1) -> dict[str, Any]:
        if ticks < 1:
            raise SimulationError("ticks must be >= 1")
        with self.lock:
            for _ in range(ticks):
                self.sim_time_seconds += self.config.simulation.tick_seconds
                self._expire_faults()
                self._recalculate()
                self._record_event("sim_step", "Simulation advanced")
                self._capture_canonical_telemetry("sim_step")
                self._record_metric_history_sample()
            return self.state_summary()

    def _record_metric_history_sample(self) -> None:
        summary = self.state_summary()
        self._metric_history_points.extend(
            metric_points_from_runtime_state(
                summary,
                self.racks,
                self.cooling_units,
                self.tenants,
                self.sim_time_seconds,
            )
        )
        if len(self._metric_history_points) > MAX_RECORDED_METRIC_POINTS:
            overflow = len(self._metric_history_points) - MAX_RECORDED_METRIC_POINTS
            self._metric_history_points = self._metric_history_points[overflow:]

    def _record_event(
        self,
        event_type: str,
        message: str,
        details: dict[str, Any] | None = None,
        level: int = logging.INFO,
    ) -> None:
        self._event_counter += 1
        event = {
            "sequence_id": self._event_counter,
            "episode_id": self._episode_id,
            "event_type": event_type,
            "sim_time_seconds": self.sim_time_seconds,
            "message": message,
            "details": deepcopy(details or {}),
        }
        self._event_log.append(event)
        if len(self._event_log) > MAX_RECORDED_EVENTS:
            self._event_log = self._event_log[-MAX_RECORDED_EVENTS:]
        if hasattr(self, "_telemetry_recorder"):
            self._telemetry_recorder.record_event(event)
        log_event(
            self.logger,
            event_type,
            self.sim_time_seconds,
            message,
            details,
            level=level,
            episode_id=self._episode_id,
        )

    @property
    def episode_id(self) -> str:
        return self._episode_id

    def recent_events(self, limit: int = 50) -> list[dict[str, Any]]:
        with self.lock:
            bounded_limit = max(0, min(int(limit), MAX_RECORDED_EVENTS))
            if bounded_limit == 0:
                return []
            return deepcopy(self._event_log[-bounded_limit:])

    def canonical_snapshot(
        self,
        query_time_seconds: int | float | None = None,
        query_watermark_sequence: str | None = None,
        lookback_seconds: int | float = 300,
        channels: list[str] | tuple[str, ...] | set[str] | None = None,
        include_config: bool = True,
        log_limit: int | None = None,
    ) -> dict[str, Any]:
        """Return the shared causal telemetry source for every inference method.

        The legacy ``observation()`` interface remains unchanged.  StateBundle
        and comparison methods can negotiate this versioned representation
        explicitly, and all receive the same immutable snapshot.
        """
        with self.lock:
            query_time = (
                float(self.sim_time_seconds)
                if query_time_seconds is None
                else float(query_time_seconds)
            )
            if query_time < 0 or query_time > self.sim_time_seconds:
                raise SimulationError(
                    "query_time_seconds must be within the current episode history"
                )
            if channels is not None:
                unsupported = set(channels) - set(CANONICAL_TELEMETRY_CHANNELS)
                if unsupported:
                    raise SimulationError(
                        f"unsupported canonical telemetry channel(s): {sorted(unsupported)}"
                    )
            if not isinstance(include_config, bool):
                raise SimulationError("include_config must be a boolean")
            if (
                log_limit is not None
                and (
                    isinstance(log_limit, bool)
                    or not isinstance(log_limit, int)
                    or log_limit < 0
                )
            ):
                raise SimulationError(
                    "log_limit must be a non-negative integer or null"
                )
            try:
                snapshot = self._telemetry_recorder.snapshot(
                    query_time_seconds=query_time,
                    query_watermark_sequence=query_watermark_sequence,
                    lookback_seconds=lookback_seconds,
                    channels=channels,
                    include_config=include_config,
                    log_limit=log_limit,
                )
                validate_canonical_snapshot(snapshot)
                return snapshot
            except ValueError as error:
                raise SimulationError(str(error)) from error

    def _capture_canonical_telemetry(self, reason: str) -> None:
        """Capture operational effects without passing oracle fault objects."""
        self._telemetry_recorder.capture(self, reason=reason)

    def _expire_faults(self) -> None:
        expired = [fault_id for fault_id, fault in self.active_faults.items() if fault.expired(self.sim_time_seconds)]
        for fault_id in expired:
            fault = self.active_faults.pop(fault_id)
            fault.status = "expired"
            self._finish_load_balancer_fault(fault)
            self._record_event(
                "fault_expired",
                "Fault expired",
                self._fault_telemetry(fault),
            )

    def start_workload(self, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        with self.lock:
            payload = payload or {}
            snapshot = self._snapshot_workload_state()
            try:
                self._update_workload(payload)
                self.workload.running = True
                self._mark_workload_job_started(force=True)
                if self._multi_tenant_enabled:
                    if "tenants" not in payload and "tenant_id" not in payload:
                        for tenant in self.tenants.values():
                            tenant.running = True
                            self._mark_tenant_job_started(tenant, force=True)
                    else:
                        for tenant in self.tenants.values():
                            if tenant.running:
                                self._mark_tenant_job_started(tenant)
                self._recalculate()
            except SimulationError:
                self._restore_workload_state(snapshot)
                raise
            self._record_event("workload_started", "Workload started", self.workload.to_dict())
            self._capture_canonical_telemetry("workload_started")
            return self.workload.to_dict()

    def update_workload(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            snapshot = self._snapshot_workload_state()
            try:
                self._update_workload(payload)
                self._recalculate()
            except SimulationError:
                self._restore_workload_state(snapshot)
                raise
            self._record_event("workload_updated", "Workload updated", self.workload.to_dict())
            self._capture_canonical_telemetry("workload_updated")
            return self.workload.to_dict()

    def stop_workload(self) -> dict[str, Any]:
        with self.lock:
            self.workload.running = False
            self.workload.queue_length = 0
            for tenant in self.tenants.values():
                tenant.running = False
                tenant.queue_length = 0
            self._recalculate()
            self._record_event("workload_stopped", "Workload stopped")
            self._capture_canonical_telemetry("workload_stopped")
            return self.workload.to_dict()

    def _update_workload(self, payload: dict[str, Any]) -> None:
        runtime_fields = {
            "queue_length",
            "running",
            "service_capacity_requests_per_second",
            "network_demand_mbps",
            "network_congestion_ratio",
            "network_latency_penalty_ms",
            "network_packet_loss_percent",
            "network_retransmit_rate",
            "network_error_rate",
            "affected_rack_id",
            "storage_demand_iops",
            "storage_utilization_ratio",
            "storage_latency_penalty_ms",
            "application_error_rate_percent",
            "dropped_requests_per_second",
            "gpu_utilization_percent",
            "queueing_latency_ms",
            "service_time_latency_ms",
            "fault_latency_penalty_ms",
            "scheduler_api_latency_ms",
            "scheduler_pending_operations",
            "average_latency_ms",
            "p95_latency_ms",
            "tenants",
            "active_tenant_count",
            "job_started_at_sim_time_seconds",
            "job_completed",
            "trace_replay_enabled",
            "trace_replay_progress",
            "current_demand_per_second",
            "uncapped_demand_per_second",
            "current_profile_type",
            "active_workload_class",
            "class_resource_demand",
            "cpu_demand",
            "memory_demand",
            "network_demand",
            "storage_demand",
            "gpu_demand",
            "maintenance_window_active",
            "maintenance_affected_server_ids",
            "maintenance_affected_server_workload_fraction",
            "allocated_server_ids",
            "allocation_weights",
            "desired_allocated_server_ids",
            "desired_allocation_weights",
            "placement_policy_status",
            "placement_policy_target_rack_id",
            "placement_policy_violating_racks",
            "workload_placement_imbalance_ratio",
            "autoscaler_current_capacity_units",
            "autoscaler_effective_server_limit",
            "autoscaler_last_scale_action_time",
            "autoscaler_status",
            "metrics_last_updated_sim_time_seconds",
            "logs_last_updated_sim_time_seconds",
            "telemetry_lag_seconds",
            "metrics_missing_ratio",
            "logs_missing_ratio",
            "telemetry_pipeline_status",
            "load_balancer_request_skew_ratio",
            "load_balancer_unhealthy_routing_fraction",
            "load_balancer_error_rate_percent",
        }
        if "profile_type" in payload and "workload_profile_type" not in payload:
            payload = {**payload, "workload_profile_type": payload["profile_type"]}
        if "profile_parameters" in payload and "workload_profile_parameters" not in payload:
            payload = {**payload, "workload_profile_parameters": payload["profile_parameters"]}
        if "job_type" in payload and "workload_class" not in payload:
            payload = {**payload, "workload_class": payload["job_type"]}
        if "workload_class_type" in payload and "workload_class" not in payload:
            payload = {**payload, "workload_class": payload["workload_class_type"]}
        if "duration_seconds" in payload and "job_duration_seconds" not in payload:
            payload = {**payload, "job_duration_seconds": payload["duration_seconds"]}
        if "trace_path" in payload and "trace_replay" not in payload:
            payload = {**payload, "trace_replay": {"path": payload["trace_path"], "format": payload.get("trace_format")}}
        if "tenants" in payload:
            self._configure_tenants(payload["tenants"])
        if "tenant_id" in payload and "tenants" not in payload and self._multi_tenant_enabled:
            tenant = self._require_tenant(payload["tenant_id"], "tenant workload update")
            self._update_tenant(tenant.tenant_id, payload)
            self._sync_tenant_state()
            return
        reset_allocation = any(key in payload for key in ("placement_strategy", "target_rack_id"))
        allowed = set(self.workload.to_dict().keys()) - runtime_fields
        for key, value in payload.items():
            if key in allowed:
                setattr(self.workload, key, value)
        if reset_allocation:
            self._clear_workload_allocation()
        self.workload.job_duration_seconds = _optional_int(self.workload.job_duration_seconds)
        if payload.get("running") is True:
            self._mark_workload_job_started(force=True)
        if self.workload.placement_strategy not in {"spread", "rack_hotspot", "random"}:
            raise SimulationError(f"unsupported placement_strategy: {self.workload.placement_strategy}")
        if self.workload.placement_strategy == "rack_hotspot" and self.workload.target_rack_id:
            self._require_rack(self.workload.target_rack_id)
        try:
            self._workload_class = build_workload_class(self.workload.workload_class)
        except ValueError as error:
            raise SimulationError(str(error)) from error
        self._apply_workload_class_state()
        self._configure_workload_trace(self.workload.trace_replay)
        if not self.workload.trace_replay_enabled:
            try:
                self._workload_profile = build_workload_profile(
                    self.workload.workload_profile_type,
                    self.workload.request_rate_per_second,
                    self.workload.workload_profile_parameters,
                )
            except ValueError as error:
                raise SimulationError(str(error)) from error
            self.workload.workload_profile_type = self._workload_profile.profile_type
            self.workload.current_profile_type = self._workload_profile.profile_type
        for server_id in self.workload.workload_profile_parameters.get("affected_server_ids", []):
            self._require_server(server_id)

    def _apply_workload_class_state(self) -> None:
        profile = self._workload_class
        self.workload.workload_class = profile.workload_class
        self.workload.active_workload_class = profile.workload_class
        self.workload.class_resource_demand = profile.resource_demand()
        self.workload.cpu_demand = profile.cpu_demand
        self.workload.memory_demand = profile.memory_demand
        self.workload.network_demand = profile.network_demand
        self.workload.storage_demand = profile.storage_demand
        self.workload.gpu_demand = profile.gpu_demand

    def _configure_workload_trace(self, trace_config: dict[str, Any] | None) -> None:
        if not trace_config:
            self._workload_trace = None
            self.workload.trace_replay_enabled = False
            self.workload.trace_replay_progress = {"enabled": False}
            return
        try:
            self._workload_trace = TraceReplayEngine.from_config(trace_config)
        except TraceReplayError as error:
            raise SimulationError(str(error)) from error
        self.workload.trace_replay = dict(trace_config)
        self.workload.trace_replay_enabled = True
        self.workload.workload_profile_type = "trace_replay"
        self.workload.current_profile_type = "trace_replay"
        self.workload.trace_replay_progress = self._workload_trace.progress(self.sim_time_seconds).to_dict()
        if self._workload_trace.tenant_ids() and not self.tenants:
            self._configure_tenants(
                [
                    {
                        "tenant_id": tenant_id,
                        "request_rate_per_second": 0.0,
                        "workload_class": "web_service",
                    }
                    for tenant_id in self._workload_trace.tenant_ids()
                ]
            )

    def _apply_workload_trace_record(self, record: TraceReplayRecord | None) -> GeneratedWorkloadDemand:
        if record is None:
            generated_demand = self._apply_workload_throttle(
                GeneratedWorkloadDemand(profile_type="trace_replay", request_rate_per_second=0.0)
            )
            self.workload.current_demand_per_second = generated_demand.request_rate_per_second
            self.workload.current_profile_type = "trace_replay"
            self.workload.trace_replay_progress = (
                self._workload_trace.progress(self.sim_time_seconds).to_dict()
                if self._workload_trace
                else {"enabled": False}
            )
            return generated_demand
        try:
            self._workload_class = build_workload_class(record.workload_class)
        except ValueError as error:
            raise SimulationError(str(error)) from error
        self._apply_workload_class_state()
        self._apply_resource_overrides_to_workload(record.resource_overrides())
        self.workload.current_profile_type = "trace_replay"
        self.workload.trace_replay_progress = (
            self._workload_trace.progress(self.sim_time_seconds).to_dict()
            if self._workload_trace
            else {"enabled": False}
        )
        generated_demand = self._apply_workload_throttle(
            GeneratedWorkloadDemand(
                profile_type="trace_replay",
                request_rate_per_second=record.request_rate_per_second,
            )
        )
        self.workload.current_demand_per_second = generated_demand.request_rate_per_second
        return generated_demand

    def _apply_workload_throttle(self, generated_demand: GeneratedWorkloadDemand) -> GeneratedWorkloadDemand:
        self.workload.uncapped_demand_per_second = generated_demand.request_rate_per_second
        capped_rate = self._capped_request_rate(
            generated_demand.request_rate_per_second,
            self.workload.throttle_rate_per_second,
        )
        if capped_rate == generated_demand.request_rate_per_second:
            return generated_demand
        return GeneratedWorkloadDemand(
            profile_type=generated_demand.profile_type,
            request_rate_per_second=capped_rate,
            affected_server_ids=set(generated_demand.affected_server_ids),
            affected_server_workload_fraction=generated_demand.affected_server_workload_fraction,
            maintenance_active=generated_demand.maintenance_active,
        )

    def _apply_tenant_throttle(
        self,
        tenant: TenantWorkloadState,
        generated_demand: GeneratedWorkloadDemand,
    ) -> GeneratedWorkloadDemand:
        tenant.uncapped_demand_per_second = generated_demand.request_rate_per_second
        capped_rate = self._capped_request_rate(
            generated_demand.request_rate_per_second,
            tenant.throttle_rate_per_second,
        )
        if capped_rate == generated_demand.request_rate_per_second:
            tenant.current_demand_per_second = generated_demand.request_rate_per_second
            return generated_demand
        capped_demand = GeneratedWorkloadDemand(
            profile_type=generated_demand.profile_type,
            request_rate_per_second=capped_rate,
            affected_server_ids=set(generated_demand.affected_server_ids),
            affected_server_workload_fraction=generated_demand.affected_server_workload_fraction,
            maintenance_active=generated_demand.maintenance_active,
        )
        tenant.current_demand_per_second = capped_demand.request_rate_per_second
        return capped_demand

    def _capped_request_rate(self, raw_rate: float, throttle_rate: float | None) -> float:
        if throttle_rate is None:
            return raw_rate
        return min(raw_rate, max(0.0, throttle_rate))

    def _apply_resource_overrides_to_workload(self, overrides: dict[str, float]) -> None:
        for resource_name, value in overrides.items():
            setattr(self.workload, f"{resource_name}_demand", value)
        self.workload.class_resource_demand = {
            "cpu": self.workload.cpu_demand,
            "memory": self.workload.memory_demand,
            "network": self.workload.network_demand,
            "storage": self.workload.storage_demand,
            "gpu": self.workload.gpu_demand,
        }

    def _configure_tenants(self, tenant_payloads: list[dict[str, Any]] | dict[str, dict[str, Any]]) -> None:
        items = (
            [{**payload, "tenant_id": tenant_id} for tenant_id, payload in tenant_payloads.items()]
            if isinstance(tenant_payloads, dict)
            else list(tenant_payloads)
        )
        self.tenants = {}
        self._tenant_profiles = {}
        self._tenant_classes = {}
        self._tenant_traces = {}
        for index, payload in enumerate(items, start=1):
            tenant = self._build_tenant_state(payload, fallback_tenant_id=f"tenant-{index}")
            self.tenants[tenant.tenant_id] = tenant
            self._rebuild_tenant_models(tenant)
        self._multi_tenant_enabled = bool(self.tenants)
        if self._multi_tenant_enabled:
            self._clear_workload_allocation()
        self._sync_tenant_state()

    def _build_tenant_state(self, payload: dict[str, Any], fallback_tenant_id: str) -> TenantWorkloadState:
        data = self._normalized_tenant_payload(payload)
        tenant = TenantWorkloadState(
            tenant_id=str(data.get("tenant_id") or data.get("id") or data.get("name") or fallback_tenant_id),
            running=bool(data.get("running", True)),
            job_duration_seconds=_optional_int(data.get("job_duration_seconds")),
            request_rate_per_second=float(data.get("request_rate_per_second", self.workload.request_rate_per_second)),
            workload_class=str(data.get("workload_class", self.workload.workload_class)),
            workload_profile_type=str(data.get("workload_profile_type", self.workload.workload_profile_type)),
            workload_profile_parameters=dict(data.get("workload_profile_parameters", {})),
            trace_replay=dict(data.get("trace_replay", {})),
            priority=int(data.get("priority", 1)),
            quota_requests_per_second=_optional_float(data.get("quota_requests_per_second")),
            max_server_count=_optional_int(data.get("max_server_count")),
            placement_strategy=str(data.get("placement_strategy", self.workload.placement_strategy)),
            target_rack_id=data.get("target_rack_id"),
        )
        if tenant.running:
            self._mark_tenant_job_started(tenant, force=True)
        self._validate_tenant_placement(tenant)
        return tenant

    def _normalized_tenant_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        data = dict(payload)
        if "profile_type" in data and "workload_profile_type" not in data:
            data["workload_profile_type"] = data["profile_type"]
        if "profile_parameters" in data and "workload_profile_parameters" not in data:
            data["workload_profile_parameters"] = data["profile_parameters"]
        if "job_type" in data and "workload_class" not in data:
            data["workload_class"] = data["job_type"]
        if "workload_class_type" in data and "workload_class" not in data:
            data["workload_class"] = data["workload_class_type"]
        if "duration_seconds" in data and "job_duration_seconds" not in data:
            data["job_duration_seconds"] = data["duration_seconds"]
        if "trace_path" in data and "trace_replay" not in data:
            data["trace_replay"] = {"path": data["trace_path"], "format": data.get("trace_format")}
        quota = data.get("quota")
        if isinstance(quota, dict):
            if "request_rate_per_second" in quota and "quota_requests_per_second" not in data:
                data["quota_requests_per_second"] = quota["request_rate_per_second"]
            if "max_server_count" in quota and "max_server_count" not in data:
                data["max_server_count"] = quota["max_server_count"]
        elif quota is not None and "quota_requests_per_second" not in data:
            data["quota_requests_per_second"] = quota
        placement = data.get("placement_constraints")
        if isinstance(placement, dict):
            if "placement_strategy" in placement and "placement_strategy" not in data:
                data["placement_strategy"] = placement["placement_strategy"]
            if "target_rack_id" in placement and "target_rack_id" not in data:
                data["target_rack_id"] = placement["target_rack_id"]
            if "max_server_count" in placement and "max_server_count" not in data:
                data["max_server_count"] = placement["max_server_count"]
        return data

    def _update_tenant(self, tenant_id: str, payload: dict[str, Any]) -> None:
        tenant = self.tenants[tenant_id]
        data = self._normalized_tenant_payload(payload)
        was_running = tenant.running
        reset_allocation = any(key in data for key in ("placement_strategy", "target_rack_id", "max_server_count"))
        update_fields = {
            "running",
            "job_duration_seconds",
            "request_rate_per_second",
            "workload_class",
            "workload_profile_type",
            "workload_profile_parameters",
            "trace_replay",
            "priority",
            "quota_requests_per_second",
            "max_server_count",
            "placement_strategy",
            "target_rack_id",
        }
        for key in update_fields:
            if key in data:
                setattr(tenant, key, data[key])
        tenant.request_rate_per_second = float(tenant.request_rate_per_second)
        tenant.running = bool(tenant.running)
        tenant.job_duration_seconds = _optional_int(tenant.job_duration_seconds)
        tenant.priority = int(tenant.priority)
        tenant.quota_requests_per_second = _optional_float(tenant.quota_requests_per_second)
        tenant.max_server_count = _optional_int(tenant.max_server_count)
        if tenant.running and (not was_running or "job_duration_seconds" in data):
            self._mark_tenant_job_started(tenant, force=True)
        self._validate_tenant_placement(tenant)
        if reset_allocation or (tenant.running and not was_running):
            self._clear_tenant_allocation(tenant)
        self._rebuild_tenant_models(tenant)

    def _validate_tenant_placement(self, tenant: TenantWorkloadState) -> None:
        if tenant.placement_strategy not in {"spread", "rack_hotspot", "random"}:
            raise SimulationError(f"unsupported placement_strategy for tenant {tenant.tenant_id}: {tenant.placement_strategy}")
        if tenant.placement_strategy == "rack_hotspot" and tenant.target_rack_id:
            self._require_rack(tenant.target_rack_id)

    def _rebuild_tenant_models(self, tenant: TenantWorkloadState) -> None:
        try:
            tenant_class = build_workload_class(tenant.workload_class)
        except ValueError as error:
            raise SimulationError(str(error)) from error
        tenant_profile = None
        if tenant.trace_replay:
            try:
                self._tenant_traces[tenant.tenant_id] = TraceReplayEngine.from_config(tenant.trace_replay)
            except TraceReplayError as error:
                raise SimulationError(str(error)) from error
            self._tenant_profiles.pop(tenant.tenant_id, None)
            tenant.trace_replay_enabled = True
            tenant.trace_replay_progress = self._tenant_traces[tenant.tenant_id].progress(
                self.sim_time_seconds,
            ).to_dict()
            tenant.workload_profile_type = "trace_replay"
            tenant.current_profile_type = "trace_replay"
        else:
            self._tenant_traces.pop(tenant.tenant_id, None)
            tenant.trace_replay_enabled = False
            tenant.trace_replay_progress = {"enabled": False}
            try:
                tenant_profile = build_workload_profile(
                    tenant.workload_profile_type,
                    tenant.request_rate_per_second,
                    tenant.workload_profile_parameters,
                )
            except ValueError as error:
                raise SimulationError(str(error)) from error
        self._tenant_classes[tenant.tenant_id] = tenant_class
        if tenant_profile is not None:
            self._tenant_profiles[tenant.tenant_id] = tenant_profile
        tenant.workload_class = tenant_class.workload_class
        tenant.active_workload_class = tenant_class.workload_class
        tenant.class_resource_demand = tenant_class.resource_demand()
        tenant.cpu_demand = tenant_class.cpu_demand
        tenant.memory_demand = tenant_class.memory_demand
        tenant.network_demand = tenant_class.network_demand
        tenant.storage_demand = tenant_class.storage_demand
        tenant.gpu_demand = tenant_class.gpu_demand
        if tenant_profile is not None:
            tenant.workload_profile_type = tenant_profile.profile_type
            tenant.current_profile_type = tenant_profile.profile_type

    def _sync_tenant_state(self) -> None:
        self.workload.tenants = {tenant_id: tenant.to_dict() for tenant_id, tenant in sorted(self.tenants.items())}
        self.workload.active_tenant_count = len([tenant for tenant in self.tenants.values() if tenant.running])

    def _snapshot_workload_state(self) -> dict[str, Any]:
        return {
            "workload": deepcopy(self.workload),
            "tenants": deepcopy(self.tenants),
            "tenant_profiles": deepcopy(self._tenant_profiles),
            "tenant_classes": deepcopy(self._tenant_classes),
            "workload_trace": deepcopy(self._workload_trace),
            "tenant_traces": deepcopy(self._tenant_traces),
            "multi_tenant_enabled": self._multi_tenant_enabled,
            "workload_class": deepcopy(self._workload_class),
            "workload_profile": deepcopy(self._workload_profile),
            "last_queue_update_sim_time": self._last_queue_update_sim_time,
        }

    def _restore_workload_state(self, snapshot: dict[str, Any]) -> None:
        self.workload = snapshot["workload"]
        self.tenants = snapshot["tenants"]
        self._tenant_profiles = snapshot["tenant_profiles"]
        self._tenant_classes = snapshot["tenant_classes"]
        self._workload_trace = snapshot["workload_trace"]
        self._tenant_traces = snapshot["tenant_traces"]
        self._multi_tenant_enabled = snapshot["multi_tenant_enabled"]
        self._workload_class = snapshot["workload_class"]
        self._workload_profile = snapshot["workload_profile"]
        self._last_queue_update_sim_time = snapshot["last_queue_update_sim_time"]

    def _clear_workload_allocation(self) -> None:
        self.workload.allocated_server_ids = []
        self.workload.allocation_weights = {}
        self.workload.desired_allocated_server_ids = []
        self.workload.desired_allocation_weights = {}

    def _clear_tenant_allocation(self, tenant: TenantWorkloadState) -> None:
        tenant.allocated_server_ids = []
        tenant.allocation_weights = {}
        tenant.desired_allocated_server_ids = []
        tenant.desired_allocation_weights = {}

    def _mark_workload_job_started(self, force: bool = False) -> None:
        if force or self.workload.job_started_at_sim_time_seconds is None or self.workload.job_completed:
            self.workload.job_started_at_sim_time_seconds = self.sim_time_seconds
            self.workload.job_completed = False

    def _mark_tenant_job_started(self, tenant: TenantWorkloadState, force: bool = False) -> None:
        if force or tenant.job_started_at_sim_time_seconds is None or tenant.job_completed:
            tenant.job_started_at_sim_time_seconds = self.sim_time_seconds
            tenant.job_completed = False

    def _expire_workload_jobs(self) -> None:
        if self._job_duration_elapsed(self.workload.job_started_at_sim_time_seconds, self.workload.job_duration_seconds):
            if self.workload.running:
                self.workload.running = False
                self.workload.queue_length = 0
                self.workload.job_completed = True
                self._clear_workload_allocation()
                for tenant in self.tenants.values():
                    tenant.running = False
                    tenant.queue_length = 0
                    self._clear_tenant_allocation(tenant)
                    tenant.job_completed = True
                    self._reset_tenant_runtime(tenant)
                self._record_event("workload_job_completed", "Workload job duration elapsed")
        for tenant in self.tenants.values():
            if self._job_duration_elapsed(tenant.job_started_at_sim_time_seconds, tenant.job_duration_seconds):
                if tenant.running:
                    tenant.running = False
                    tenant.queue_length = 0
                    self._clear_tenant_allocation(tenant)
                    tenant.job_completed = True
                    self._record_event(
                        "tenant_job_completed",
                        "Tenant workload job duration elapsed",
                        {"tenant_id": tenant.tenant_id},
                    )
        self._sync_tenant_state()

    def _job_duration_elapsed(self, started_at: int | None, duration_seconds: int | None) -> bool:
        return (
            started_at is not None
            and duration_seconds is not None
            and duration_seconds >= 0
            and self.sim_time_seconds - started_at >= duration_seconds
        )

    def inject_fault(self, request: FaultRequest) -> dict[str, Any]:
        with self.lock:
            if request.fault_type not in SUPPORTED_FAULTS:
                raise SimulationError(f"unsupported fault_type: {request.fault_type}")
            self._validate_fault_target(request)
            self._fault_counter += 1
            fault = Fault(
                fault_id=f"fault-{self._fault_counter:04d}",
                fault_type=request.fault_type,
                target=request.target,
                severity=request.severity,
                duration_seconds=request.duration_seconds,
                started_at_sim_time_seconds=self.sim_time_seconds,
                parameters={
                    key: value
                    for key, value in {
                        "period_seconds": request.period_seconds,
                        "duty_cycle": request.duty_cycle,
                        "failure_window_seconds": request.failure_window_seconds,
                    }.items()
                    if value is not None
                },
            )
            if request.fault_type == "power_budget_violation":
                rack = self._require_rack(request.target)
                current_draw_kw = sum(server.power_kw for server in rack.servers)
                idle_draw_kw = sum(
                    self.config.server.idle_power_kw
                    for server in rack.servers
                    if server.status != "failed"
                )
                dynamic_draw_kw = max(0.0, current_draw_kw - idle_draw_kw)
                if dynamic_draw_kw > 0.0:
                    injected_budget_kw = idle_draw_kw + dynamic_draw_kw * max(
                        0.05,
                        1.0 - request.severity,
                    )
                else:
                    injected_budget_kw = current_draw_kw / (
                        1.0 + request.severity
                    )
                fault.parameters["injected_power_budget_kw"] = max(
                    0.0001,
                    round(injected_budget_kw, 4),
                )
            if request.fault_type == "load_balancer_misconfiguration":
                self._begin_load_balancer_fault()
            self.active_faults[fault.fault_id] = fault
            self._recalculate()
            self._record_event(
                "fault_injected",
                f"{request.fault_type} injected",
                self._fault_telemetry(fault),
            )
            self._capture_canonical_telemetry("state_transition")
            return fault.to_dict()

    def _validate_fault_target(self, request: FaultRequest) -> None:
        if request.fault_type == "cooling_degradation":
            self._require_cooling_unit(request.target)
        elif request.fault_type in {"rack_hotspot", "power_overload", "power_budget_violation"}:
            self._require_rack(request.target)
        elif request.fault_type in {"server_failure", "intermittent_server_failure"}:
            self._require_server(request.target)
        elif request.fault_type in {"thermal_sensor_miscalibration", "thermal_throttling"}:
            if request.target.startswith("rack-"):
                self._require_rack(request.target)
            elif request.target.startswith("server-"):
                self._require_server(request.target)
            else:
                raise SimulationError(f"{request.fault_type} target must be a rack or server")
        elif request.fault_type in {"network_partition", "tor_packet_loss"}:
            self._require_rack(request.target)
        elif request.fault_type == "network_congestion_burst":
            if request.target.startswith("rack-"):
                self._require_rack(request.target)
            elif request.target in APPLICATION_TARGETS:
                return
            elif self.tenants and request.target in self.tenants:
                return
            elif request.target == self.workload.tenant_id:
                return
            else:
                raise SimulationError(
                    "network_congestion_burst target must be workload, application, global, an existing tenant, or a rack"
                )
        elif request.fault_type == "storage_io_saturation":
            if request.target not in STORAGE_TARGETS:
                raise SimulationError(f"storage_io_saturation target must be one of {sorted(STORAGE_TARGETS)}")
        elif request.fault_type == "control_plane_degradation":
            if request.target not in CONTROL_PLANE_TARGETS:
                raise SimulationError(
                    f"control_plane_degradation target must be one of {sorted(CONTROL_PLANE_TARGETS)}"
                )
        elif request.fault_type == "autoscaler_misconfiguration":
            if request.target not in CONTROL_PLANE_TARGETS | {"autoscaler", "workload"}:
                raise SimulationError(
                    "autoscaler_misconfiguration target must be autoscaler, workload, or control-plane scoped"
                )
        elif request.fault_type == "monitoring_pipeline_failure":
            if request.target not in MONITORING_TARGETS:
                raise SimulationError(
                    f"monitoring_pipeline_failure target must be one of {sorted(MONITORING_TARGETS)}"
                )
        elif request.fault_type == "placement_policy_misconfiguration":
            if request.target.startswith("rack-"):
                self._require_rack(request.target)
            elif request.target not in CONTROL_PLANE_TARGETS | {"scheduler", "placement-policy", "workload"}:
                raise SimulationError(
                    "placement_policy_misconfiguration target must be a rack, scheduler, workload, or control-plane scoped"
                )
        elif request.fault_type == "load_balancer_misconfiguration":
            if request.target not in APPLICATION_TARGETS | {"load-balancer", "frontend", "service"}:
                raise SimulationError(
                    "load_balancer_misconfiguration target must be application, workload, or load-balancer scoped"
                )
        elif request.fault_type == "application_error":
            if request.target in APPLICATION_TARGETS:
                return
            if self.tenants:
                if request.target in self.tenants:
                    return
            elif request.target == self.workload.tenant_id:
                return
            raise SimulationError(
                "application_error target must be application, workload, global, or an existing tenant"
            )

    def remove_fault(self, fault_id: str) -> dict[str, Any]:
        with self.lock:
            if fault_id not in self.active_faults:
                raise SimulationError(f"fault not found: {fault_id}")
            fault = self.active_faults.pop(fault_id)
            fault.status = "removed"
            self._finish_load_balancer_fault(fault)
            self._recalculate()
            self._record_event(
                "fault_removed",
                "Fault removed",
                self._fault_telemetry(fault),
            )
            self._capture_canonical_telemetry("state_transition")
            return fault.to_dict()

    def clear_faults(self) -> list[dict[str, Any]]:
        with self.lock:
            removed = []
            for fault_id in list(self.active_faults):
                removed.append(self.remove_fault(fault_id))
            return removed

    def apply_control(self, request: ControlRequest) -> dict[str, Any]:
        with self.lock:
            if request.action_type not in SUPPORTED_ACTIONS:
                raise SimulationError(f"unsupported action_type: {request.action_type}")
            details = self._apply_control_effect(request)
            if (
                request.action_type == "update_load_balancer_config"
                and self._load_balancer_fault_baseline is not None
            ):
                self._load_balancer_fault_baseline = (
                    self._load_balancer_configuration_state()
                )
            self._action_counter += 1
            action = ControlAction(
                action_id=f"action-{self._action_counter:04d}",
                action_type=request.action_type,
                status="applied",
                sim_time_seconds=self.sim_time_seconds,
                details=details,
            )
            self.controls.append(action)
            self._recalculate()
            self._record_event(
                "control_applied",
                f"{request.action_type} applied",
                {"action_id": action.action_id, "action_type": action.action_type, **details},
            )
            self._capture_canonical_telemetry("control_applied")
            return action.to_dict()

    def _apply_control_effect(self, request: ControlRequest) -> dict[str, Any]:
        if request.action_type == "calibrate_sensor":
            target = request.target or request.server_id
            if not target:
                raise SimulationError("calibrate_sensor requires target")
            if request.calibration_offset_c is None and request.mark_untrusted is None:
                raise SimulationError("calibrate_sensor requires calibration_offset_c or mark_untrusted")
            if target.startswith("rack-"):
                self._require_rack(target)
            elif target.startswith("server-"):
                self._require_server(target)
            else:
                raise SimulationError("calibrate_sensor target must be a rack or server")
            return {
                "target": target,
                "calibration_offset_c": request.calibration_offset_c,
                "mark_untrusted": request.mark_untrusted,
            }
        if request.action_type == "set_cooling":
            target = request.target or "cooling-unit-1"
            unit = self._require_cooling_unit(target)
            if request.fan_speed_percent is None and request.supply_air_temperature_c is None:
                raise SimulationError("set_cooling requires fan_speed_percent or supply_air_temperature_c")
            if request.fan_speed_percent is not None:
                unit.fan_speed_percent = request.fan_speed_percent
            if request.supply_air_temperature_c is not None:
                unit.supply_air_temperature_c = request.supply_air_temperature_c
            return {"target": target}
        if request.action_type == "migrate_workload":
            if not request.source_rack_id or not request.target_rack_id or request.workload_fraction is None:
                raise SimulationError("migrate_workload requires source_rack_id, target_rack_id, workload_fraction")
            self._require_rack(request.source_rack_id)
            self._require_rack(request.target_rack_id)
            if self._multi_tenant_enabled:
                tenant = self._require_tenant(request.tenant_id, "migrate_workload")
                self._migrate_tenant_allocation(
                    tenant,
                    request.source_rack_id,
                    request.target_rack_id,
                    request.workload_fraction,
                )
                return {
                    "tenant_id": tenant.tenant_id,
                    "source_rack_id": request.source_rack_id,
                    "target_rack_id": request.target_rack_id,
                    "workload_fraction": request.workload_fraction,
                }
            self._migrate_workload_allocation(
                request.source_rack_id,
                request.target_rack_id,
                request.workload_fraction,
            )
            return {
                "source_rack_id": request.source_rack_id,
                "target_rack_id": request.target_rack_id,
                "workload_fraction": request.workload_fraction,
            }
        if request.action_type == "throttle_workload":
            if request.request_rate_per_second is None:
                raise SimulationError("throttle_workload requires request_rate_per_second")
            if self._multi_tenant_enabled:
                tenant = self._require_tenant(request.tenant_id, "throttle_workload")
                tenant.request_rate_per_second = request.request_rate_per_second
                tenant.throttle_rate_per_second = request.request_rate_per_second
                if tenant.workload_profile_type == "steady":
                    self._tenant_profiles[tenant.tenant_id] = build_workload_profile(
                        tenant.workload_profile_type,
                        tenant.request_rate_per_second,
                        tenant.workload_profile_parameters,
                    )
                return {
                    "tenant_id": tenant.tenant_id,
                    "request_rate_per_second": request.request_rate_per_second,
                }
            self.workload.request_rate_per_second = request.request_rate_per_second
            self.workload.throttle_rate_per_second = request.request_rate_per_second
            if self.workload.workload_profile_type == "steady":
                self._workload_profile = build_workload_profile(
                    self.workload.workload_profile_type,
                    self.workload.request_rate_per_second,
                    self.workload.workload_profile_parameters,
                )
            return {"request_rate_per_second": request.request_rate_per_second}
        if request.action_type == "update_autoscaler_policy":
            if all(
                value is None
                for value in (
                    request.min_capacity,
                    request.max_capacity,
                    request.target_utilization_percent,
                    request.cooldown_seconds,
                )
            ):
                raise SimulationError(
                    "update_autoscaler_policy requires min_capacity, max_capacity, "
                    "target_utilization_percent, or cooldown_seconds"
                )
            min_capacity = (
                self.workload.autoscaler_min_capacity
                if request.min_capacity is None
                else int(request.min_capacity)
            )
            max_capacity = (
                self.workload.autoscaler_max_capacity
                if request.max_capacity is None
                else int(request.max_capacity)
            )
            if max_capacity is not None and min_capacity > max_capacity:
                raise SimulationError("update_autoscaler_policy requires min_capacity <= max_capacity")
            self.workload.autoscaler_enabled = True
            self.workload.autoscaler_min_capacity = min_capacity
            self.workload.autoscaler_max_capacity = max_capacity
            if request.target_utilization_percent is not None:
                self.workload.autoscaler_target_utilization_percent = request.target_utilization_percent
            if request.cooldown_seconds is not None:
                self.workload.autoscaler_cooldown_seconds = int(request.cooldown_seconds)
            self.workload.autoscaler_last_scale_action_time = self.sim_time_seconds
            return {
                "min_capacity": request.min_capacity,
                "max_capacity": request.max_capacity,
                "target_utilization_percent": request.target_utilization_percent,
                "cooldown_seconds": request.cooldown_seconds,
            }
        if request.action_type == "repair_monitoring_pipeline":
            target = request.target or "monitoring-pipeline"
            return {"target": target}
        if request.action_type == "update_placement_policy":
            if request.placement_strategy is None and request.target_rack_id is None and request.forbidden_rack_ids is None and request.max_server_count is None:
                raise SimulationError(
                    "update_placement_policy requires placement_strategy, target_rack_id, "
                    "forbidden_rack_ids, or max_server_count"
                )
            if request.placement_strategy is not None and request.placement_strategy not in {"spread", "rack_hotspot", "random"}:
                raise SimulationError("update_placement_policy placement_strategy must be spread, rack_hotspot, or random")
            if request.target_rack_id:
                self._require_rack(request.target_rack_id)
            forbidden_rack_ids = (
                list(self.workload.forbidden_rack_ids)
                if request.forbidden_rack_ids is None
                else list(request.forbidden_rack_ids)
            )
            for rack_id in forbidden_rack_ids:
                self._require_rack(rack_id)
            if request.placement_strategy is not None:
                self.workload.placement_strategy = request.placement_strategy
            if request.target_rack_id is not None:
                self.workload.target_rack_id = request.target_rack_id
            self.workload.forbidden_rack_ids = forbidden_rack_ids
            if request.max_server_count is not None:
                self.workload.max_server_count = int(request.max_server_count)
            healthy_servers = [server for server in self.servers if server.status == "healthy"]
            self._set_workload_desired_allocation_weights(
                self._candidate_workload_allocation_weights(healthy_servers)
            )
            return {
                "placement_strategy": request.placement_strategy,
                "target_rack_id": request.target_rack_id,
                "forbidden_rack_ids": request.forbidden_rack_ids,
                "max_server_count": request.max_server_count,
            }
        if request.action_type == "update_load_balancer_config":
            if (
                request.backend_weights is None
                and request.remove_backend_ids is None
                and request.add_backend_ids is None
                and request.routing_policy is None
                and request.reset_to_equal_weights is None
            ):
                raise SimulationError(
                    "update_load_balancer_config requires backend_weights, remove_backend_ids, "
                    "add_backend_ids, routing_policy, or reset_to_equal_weights"
                )
            if request.routing_policy is not None and request.routing_policy not in {"round_robin", "weighted", "sticky"}:
                raise SimulationError("update_load_balancer_config routing_policy must be round_robin, weighted, or sticky")

            backend_ids = self._current_load_balancer_backend_ids()
            backend_set = set(backend_ids)
            for server_id in request.remove_backend_ids or []:
                self._require_server(server_id)
                backend_set.discard(server_id)
            for server_id in request.add_backend_ids or []:
                self._require_server(server_id)
                backend_set.add(server_id)
            if request.backend_weights:
                for server_id, weight in request.backend_weights.items():
                    self._require_server(server_id)
                    if weight < 0.0:
                        raise SimulationError("update_load_balancer_config backend_weights must be non-negative")
                    backend_set.add(server_id)

            if not backend_set:
                raise SimulationError("update_load_balancer_config requires at least one backend server")

            ordered_backend_ids = sorted(backend_set)
            if request.reset_to_equal_weights:
                backend_weights = {server_id: 1.0 for server_id in ordered_backend_ids}
            else:
                backend_weights = {
                    server_id: float(self.workload.load_balancer_backend_weights.get(server_id, 1.0))
                    for server_id in ordered_backend_ids
                }
            if request.backend_weights:
                for server_id, weight in request.backend_weights.items():
                    if server_id in backend_set:
                        backend_weights[server_id] = float(weight)

            self.workload.load_balancer_enabled = True
            self.workload.load_balancer_backend_server_ids = ordered_backend_ids
            self.workload.load_balancer_backend_weights = backend_weights
            if request.reset_to_equal_weights:
                self.workload.load_balancer_unhealthy_backend_ids = []
            else:
                removed_ids = set(request.remove_backend_ids or [])
                self.workload.load_balancer_unhealthy_backend_ids = [
                    server_id
                    for server_id in self.workload.load_balancer_unhealthy_backend_ids
                    if server_id in backend_set and server_id not in removed_ids
                ]
            if request.routing_policy is not None:
                self.workload.load_balancer_routing_policy = request.routing_policy
            elif request.reset_to_equal_weights:
                self.workload.load_balancer_routing_policy = "round_robin"

            return {
                "backend_weights": request.backend_weights,
                "remove_backend_ids": request.remove_backend_ids,
                "add_backend_ids": request.add_backend_ids,
                "routing_policy": request.routing_policy,
                "reset_to_equal_weights": request.reset_to_equal_weights,
            }
        if request.action_type == "set_server_maintenance":
            servers = self._maintenance_action_servers(request)
            for server in servers:
                server.status = "maintenance"
            return {
                "server_id": (
                    servers[0].server_id
                    if request.server_id is not None
                    else None
                ),
                "rack_id": request.rack_id,
                "affected_server_count": len(servers),
            }
        if request.action_type == "clear_server_maintenance":
            servers = self._maintenance_action_servers(request)
            for server in servers:
                if server.status == "maintenance":
                    server.status = "healthy"
            return {
                "server_id": (
                    servers[0].server_id
                    if request.server_id is not None
                    else None
                ),
                "rack_id": request.rack_id,
                "affected_server_count": len(servers),
            }
        raise SimulationError(f"unsupported action_type: {request.action_type}")

    def _maintenance_action_servers(
        self,
        request: ControlRequest,
    ) -> list[Server]:
        if bool(request.server_id) == bool(request.rack_id):
            raise SimulationError(
                "maintenance action requires exactly one of server_id or rack_id"
            )
        if request.server_id:
            return [self._require_server(request.server_id)]
        rack = self._require_rack(request.rack_id or "")
        return list(rack.servers)

    def _fault_telemetry(self, fault: Fault) -> dict[str, Any]:
        data = fault.to_dict()
        elapsed = self.sim_time_seconds - fault.started_at_sim_time_seconds
        data["remaining_duration_seconds"] = max(0, fault.duration_seconds - elapsed)
        return data

    def _network_partition_severity_by_rack(self) -> dict[str, float]:
        severities: dict[str, float] = {}
        for fault in self.active_faults.values():
            if fault.fault_type == "network_partition":
                severities[fault.target] = max(severities.get(fault.target, 0.0), fault.severity)
        return severities

    def _apply_network_partition_to_weights(self, weights: dict[str, float]) -> dict[str, float]:
        partition_severity_by_rack = self._network_partition_severity_by_rack()
        if not partition_severity_by_rack:
            return weights
        rack_id_by_server = {server.server_id: server.rack_id for server in self.servers}
        return {
            server_id: weight * max(0.0, 1.0 - partition_severity_by_rack.get(rack_id_by_server.get(server_id), 0.0))
            for server_id, weight in weights.items()
        }

    def _allocation_share_for_rack(self, weights: dict[str, float], rack_id: str) -> float:
        total_weight = sum(weight for weight in weights.values() if weight > 0.0)
        if total_weight <= 0.0:
            return 0.0
        rack_weight = 0.0
        for server_id, weight in weights.items():
            if weight <= 0.0:
                continue
            try:
                server = self._require_server(server_id)
            except SimulationError:
                continue
            if server.rack_id == rack_id:
                rack_weight += weight
        return min(1.0, max(0.0, rack_weight / total_weight))

    def _burst_profile_active(
        self,
        profile_type: str,
        profile_parameters: dict[str, Any],
        demand_rate: float,
        configured_rate: float,
    ) -> bool:
        if profile_type != "burst":
            return False
        baseline_rate = float(profile_parameters.get("baseline_rate_per_second", configured_rate))
        return demand_rate > baseline_rate * 1.05

    def _network_congestion_target_share(
        self,
        fault: Fault,
        weights: dict[str, float],
        tenant_id: str | None,
    ) -> float:
        if fault.target.startswith("rack-"):
            return self._allocation_share_for_rack(weights, fault.target)
        if fault.target in APPLICATION_TARGETS:
            return 1.0
        if tenant_id is not None:
            return 1.0 if fault.target == tenant_id else 0.0
        return 1.0 if fault.target == self.workload.tenant_id else 0.0

    def _network_congestion_capacity_factor(
        self,
        weights: dict[str, float],
        demand_rate: float,
        tenant: TenantWorkloadState | None = None,
    ) -> tuple[float, bool, str | None]:
        tenant_id = tenant.tenant_id if tenant is not None else None
        profile_type = tenant.current_profile_type if tenant is not None else self.workload.current_profile_type
        configured_profile_type = tenant.workload_profile_type if tenant is not None else self.workload.workload_profile_type
        profile_parameters = tenant.workload_profile_parameters if tenant is not None else self.workload.workload_profile_parameters
        configured_rate = tenant.request_rate_per_second if tenant is not None else self.workload.request_rate_per_second
        burst_active = self._burst_profile_active(
            configured_profile_type or profile_type,
            profile_parameters,
            demand_rate,
            configured_rate,
        )
        if not burst_active:
            return 1.0, False, None

        capacity_factor = 1.0
        affected_target: str | None = None
        for fault in self.active_faults.values():
            if fault.fault_type != "network_congestion_burst":
                continue
            target_share = self._network_congestion_target_share(fault, weights, tenant_id)
            if target_share <= 0.0:
                continue
            fault_factor = max(0.05, 1.0 - fault.severity * 0.78 * target_share)
            if fault_factor < capacity_factor:
                capacity_factor = fault_factor
                affected_target = fault.target
        return capacity_factor, capacity_factor < 0.999, affected_target

    def _tor_packet_loss_metrics_for_weights(
        self,
        weights: dict[str, float],
    ) -> tuple[float, float, float, float, str | None]:
        packet_loss_percent = 0.0
        retransmit_rate = 0.0
        network_error_rate = 0.0
        latency_penalty_ms = 0.0
        affected_rack_id: str | None = None
        for fault in self.active_faults.values():
            if fault.fault_type != "tor_packet_loss":
                continue
            rack_share = self._allocation_share_for_rack(weights, fault.target)
            if rack_share <= 0.0:
                continue
            packet_loss_percent = max(packet_loss_percent, fault.severity * 12.0 * rack_share)
            retransmit_rate = max(retransmit_rate, fault.severity * 28.0 * rack_share)
            network_error_rate = max(network_error_rate, fault.severity * 6.0 * rack_share)
            latency_penalty_ms = max(latency_penalty_ms, fault.severity * 180.0 * rack_share)
            affected_rack_id = fault.target
        return (
            round(packet_loss_percent, 4),
            round(retransmit_rate, 4),
            round(network_error_rate, 4),
            round(latency_penalty_ms, 4),
            affected_rack_id,
        )

    def _service_capacity_multiplier(self) -> float:
        severity = self._max_fault_severity("control_plane_degradation")
        return max(0.05, 1.0 - severity * 0.85)

    def _effective_storage_capacity_iops(self) -> float:
        severity = self._max_fault_severity("storage_io_saturation")
        capacity_factor = max(0.05, 1.0 - severity * 0.9)
        return self.workload.storage_capacity_iops * capacity_factor

    def _max_fault_severity(self, fault_type: str) -> float:
        return max(
            (fault.severity for fault in self.active_faults.values() if fault.fault_type == fault_type),
            default=0.0,
        )

    def _application_error_rate_percent(self, tenant_id: str | None = None, demand_rate: float = 0.0) -> float:
        if demand_rate <= 0.0:
            return 0.0
        rate = 0.0
        for fault in self.active_faults.values():
            if fault.fault_type == "application_error" and self._application_fault_applies(fault, tenant_id):
                rate = max(rate, fault.severity * 100.0)
        return round(min(100.0, rate), 4)

    def _application_fault_applies(self, fault: Fault, tenant_id: str | None = None) -> bool:
        if fault.target in APPLICATION_TARGETS:
            return True
        if tenant_id is not None:
            return fault.target == tenant_id
        return fault.target == self.workload.tenant_id

    def _latest_calibration_control(self, target: str) -> dict[str, Any] | None:
        for control in reversed(self.controls):
            if control.action_type == "calibrate_sensor" and control.details.get("target") == target:
                return control.details
        return None

    def _latest_control_for_action(self, action_type: str) -> ControlAction | None:
        for control in reversed(self.controls):
            if control.action_type == action_type:
                return control
        return None

    def _control_applied_after_fault_started(self, action_type: str, fault: Fault) -> bool:
        control = self._latest_control_for_action(action_type)
        return control is not None and control.sim_time_seconds >= fault.started_at_sim_time_seconds

    def _monitoring_pipeline_repaired_for_fault(self, fault: Fault) -> bool:
        control = self._latest_control_for_action("repair_monitoring_pipeline")
        if control is None or control.sim_time_seconds < fault.started_at_sim_time_seconds:
            return False
        return control.details.get("target") in {
            "monitoring-pipeline",
            "telemetry-pipeline",
            "metrics-pipeline",
            "monitoring",
        }

    def _apply_autoscaler_state(self, severity: float, updated: bool) -> None:
        healthy_count = len([server for server in self.servers if server.status == "healthy"])
        if severity > 0.0 and not updated:
            max_capacity = max(1, int(round(2 + (1.0 - severity) * 3)))
            self.workload.autoscaler_enabled = True
            self.workload.autoscaler_min_capacity = 1
            self.workload.autoscaler_max_capacity = max_capacity
            self.workload.autoscaler_target_utilization_percent = 98.0
            self.workload.autoscaler_cooldown_seconds = 900
            self.workload.autoscaler_status = "misconfigured"
        elif updated:
            self.workload.autoscaler_status = "updated"
        else:
            self.workload.autoscaler_status = "normal"

        configured_limit = self.workload.autoscaler_max_capacity
        if not self.workload.autoscaler_enabled or configured_limit is None:
            effective_limit = healthy_count
        else:
            effective_limit = min(healthy_count, max(self.workload.autoscaler_min_capacity, configured_limit))
        self.workload.autoscaler_effective_server_limit = effective_limit
        self.workload.autoscaler_current_capacity_units = effective_limit

    def _apply_monitoring_pipeline_state(self, severity: float, repaired: bool) -> None:
        if severity > 0.0 and not repaired:
            lag_seconds = max(5, int(round(120 * severity)))
            self.workload.telemetry_lag_seconds = lag_seconds
            self.workload.metrics_last_updated_sim_time_seconds = max(0, self.sim_time_seconds - lag_seconds)
            self.workload.logs_last_updated_sim_time_seconds = max(0, self.sim_time_seconds - lag_seconds)
            self.workload.metrics_missing_ratio = round(min(0.95, 0.75 * severity), 4)
            self.workload.logs_missing_ratio = round(min(0.95, 0.65 * severity), 4)
            self.workload.telemetry_pipeline_status = "stale"
        else:
            self.workload.metrics_last_updated_sim_time_seconds = self.sim_time_seconds
            self.workload.logs_last_updated_sim_time_seconds = self.sim_time_seconds
            self.workload.telemetry_lag_seconds = 0
            self.workload.metrics_missing_ratio = 0.0
            self.workload.logs_missing_ratio = 0.0
            self.workload.telemetry_pipeline_status = "repaired" if repaired else "normal"

    def _apply_placement_policy_state(self, target: str | None, severity: float, updated: bool) -> None:
        if target and severity > 0.0 and not updated:
            target_rack_id = target if target.startswith("rack-") else "rack-r1-row1-02"
            self._require_rack(target_rack_id)
            self.workload.placement_policy_status = "misconfigured"
            self.workload.placement_policy_target_rack_id = target_rack_id
            self.workload.placement_strategy = "rack_hotspot"
            self.workload.target_rack_id = target_rack_id
            self.workload.forbidden_rack_ids = []
            self.workload.max_server_count = max(1, int(round(2 + (1.0 - severity) * 3)))
            healthy_servers = [server for server in self.servers if server.status == "healthy"]
            self._set_workload_desired_allocation_weights(
                self._candidate_workload_allocation_weights(healthy_servers)
            )
        elif updated:
            self.workload.placement_policy_status = "updated"
        else:
            self.workload.placement_policy_status = "normal"
            self.workload.placement_policy_target_rack_id = self.workload.target_rack_id

    def _current_load_balancer_backend_ids(self) -> list[str]:
        backend_ids = [
            server_id
            for server_id in self.workload.load_balancer_backend_server_ids
            if self._server_exists(server_id)
        ]
        if backend_ids:
            return sorted(dict.fromkeys(backend_ids))

        weighted_ids = [
            server_id
            for server_id, weight in self.workload.desired_allocation_weights.items()
            if weight > 0.0 and self._server_exists(server_id)
        ]
        if weighted_ids:
            return sorted(dict.fromkeys(weighted_ids))

        healthy_ids = [server.server_id for server in self.servers if server.status == "healthy"]
        if healthy_ids:
            return sorted(healthy_ids)
        return sorted(server.server_id for server in self.servers)

    def _preferred_load_balancer_fault_backend(self, backend_ids: list[str]) -> str | None:
        preferred = "server-r1-row1-rack02-03"
        if preferred in backend_ids:
            return preferred
        return backend_ids[0] if backend_ids else None

    def _begin_load_balancer_fault(self) -> None:
        if self._load_balancer_fault_baseline is not None:
            return
        self._load_balancer_fault_baseline = (
            self._load_balancer_configuration_state()
        )

    def _load_balancer_configuration_state(self) -> dict[str, Any]:
        return {
            "backend_server_ids": list(
                self.workload.load_balancer_backend_server_ids
            ),
            "backend_weights": dict(self.workload.load_balancer_backend_weights),
            "unhealthy_backend_ids": list(
                self.workload.load_balancer_unhealthy_backend_ids
            ),
            "routing_policy": self.workload.load_balancer_routing_policy,
            "enabled": self.workload.load_balancer_enabled,
        }

    def _finish_load_balancer_fault(self, fault: Fault) -> None:
        if fault.fault_type != "load_balancer_misconfiguration":
            return
        if any(
            active.fault_type == "load_balancer_misconfiguration"
            for active in self.active_faults.values()
        ):
            return
        baseline = self._load_balancer_fault_baseline
        self._load_balancer_fault_baseline = None
        if baseline is None:
            return
        self.workload.load_balancer_enabled = bool(baseline["enabled"])
        self.workload.load_balancer_backend_server_ids = list(
            baseline["backend_server_ids"]
        )
        self.workload.load_balancer_backend_weights = dict(
            baseline["backend_weights"]
        )
        self.workload.load_balancer_unhealthy_backend_ids = list(
            baseline["unhealthy_backend_ids"]
        )
        self.workload.load_balancer_routing_policy = str(
            baseline["routing_policy"]
        )

    def _apply_load_balancer_state(self, severity: float, updated: bool) -> None:
        backend_ids = self._current_load_balancer_backend_ids()
        if not backend_ids:
            self.workload.load_balancer_enabled = False
            self.workload.load_balancer_backend_server_ids = []
            self.workload.load_balancer_backend_weights = {}
            self.workload.load_balancer_unhealthy_backend_ids = []
            self.workload.load_balancer_routing_policy = "round_robin"
            self.workload.load_balancer_request_skew_ratio = 0.0
            self.workload.load_balancer_unhealthy_routing_fraction = 0.0
            self.workload.load_balancer_error_rate_percent = 0.0
            return

        self.workload.load_balancer_enabled = True
        self.workload.load_balancer_backend_server_ids = backend_ids
        if severity > 0.0 and not updated:
            hot_backend_id = self._preferred_load_balancer_fault_backend(backend_ids)
            cold_weight = round(max(0.005, (1.0 - severity) * 0.08), 6)
            self.workload.load_balancer_backend_weights = {
                server_id: (1.0 if server_id == hot_backend_id else cold_weight)
                for server_id in backend_ids
            }
            self.workload.load_balancer_unhealthy_backend_ids = [hot_backend_id] if hot_backend_id else []
            self.workload.load_balancer_routing_policy = "sticky"
            return

        existing_weights = {
            server_id: float(self.workload.load_balancer_backend_weights.get(server_id, 1.0))
            for server_id in backend_ids
        }
        if not existing_weights or all(weight <= 0.0 for weight in existing_weights.values()):
            existing_weights = {server_id: 1.0 for server_id in backend_ids}
        self.workload.load_balancer_backend_weights = existing_weights
        if not updated:
            self.workload.load_balancer_unhealthy_backend_ids = []
            self.workload.load_balancer_routing_policy = "round_robin"

    def _apply_load_balancer_to_weights(self, weights: dict[str, float]) -> dict[str, float]:
        if not self.workload.load_balancer_enabled or not self.workload.load_balancer_backend_weights:
            return weights
        lb_backend_ids = set(self.workload.load_balancer_backend_server_ids)
        if not lb_backend_ids:
            return weights
        weighted = {
            server_id: weights.get(server_id, 0.0) * self.workload.load_balancer_backend_weights.get(server_id, 0.0)
            for server_id in weights
            if server_id in lb_backend_ids
        }
        if not any(weight > 0.0 for weight in weighted.values()):
            return weights
        return weighted

    def _update_load_balancer_runtime_metrics(self) -> None:
        backend_weights = {
            server_id: weight
            for server_id, weight in self.workload.load_balancer_backend_weights.items()
            if server_id in set(self.workload.load_balancer_backend_server_ids) and weight > 0.0
        }
        total_weight = sum(backend_weights.values())
        if not self.workload.load_balancer_enabled or total_weight <= 0.0:
            self.workload.load_balancer_request_skew_ratio = 0.0
            self.workload.load_balancer_unhealthy_routing_fraction = 0.0
            self.workload.load_balancer_error_rate_percent = 0.0
            return

        unhealthy_backend_ids = set(self.workload.load_balancer_unhealthy_backend_ids)
        unhealthy_weight = sum(
            weight
            for server_id, weight in backend_weights.items()
            if server_id in unhealthy_backend_ids
        )
        max_share = max(backend_weights.values(), default=0.0) / total_weight
        ideal_share = 1.0 / max(len(backend_weights), 1)
        skew_ratio = max(0.0, max_share - ideal_share)
        unhealthy_fraction = unhealthy_weight / total_weight
        self.workload.load_balancer_request_skew_ratio = round(skew_ratio, 4)
        self.workload.load_balancer_unhealthy_routing_fraction = round(unhealthy_fraction, 4)
        imbalance_error = max(0.0, skew_ratio - 0.2) * 20.0
        unhealthy_error = unhealthy_fraction * 35.0
        self.workload.load_balancer_error_rate_percent = round(
            min(100.0, unhealthy_error + imbalance_error),
            4,
        )

    def _calibrated_sensor_bias(self, target: str, raw_bias_c: float) -> tuple[float, str]:
        control = self._latest_calibration_control(target)
        if control is None:
            return raw_bias_c, "biased"
        if control.get("mark_untrusted") is True:
            return 0.0, "untrusted"
        calibration_offset = control.get("calibration_offset_c")
        if isinstance(calibration_offset, int | float):
            corrected = raw_bias_c + float(calibration_offset)
            if abs(corrected) <= 1.0:
                return 0.0, "calibrated"
            return corrected, "biased"
        return raw_bias_c, "biased"

    def _intermittent_failure_is_active(self, fault: Fault) -> bool:
        parameters = fault.parameters if isinstance(fault.parameters, dict) else {}
        period_seconds = int(parameters.get("period_seconds") or 6)
        if period_seconds <= 0:
            period_seconds = 6
        if parameters.get("failure_window_seconds") is not None:
            failure_window = int(parameters["failure_window_seconds"])
        else:
            duty_cycle = float(parameters.get("duty_cycle") or 0.5)
            failure_window = max(1, int(round(period_seconds * duty_cycle)))
        failure_window = max(1, min(period_seconds, failure_window))
        elapsed = max(0, self.sim_time_seconds - fault.started_at_sim_time_seconds)
        return elapsed % period_seconds < failure_window

    def _workload_allocated_to_target(self, target: str) -> bool:
        allocated_server_ids = set(self.workload.allocated_server_ids)
        if target.startswith("server-"):
            return target in allocated_server_ids
        if target.startswith("rack-"):
            return any(self._require_server(server_id).rack_id == target for server_id in allocated_server_ids)
        return False

    def _update_placement_policy_runtime_metrics(self) -> None:
        weights_by_rack: dict[str, float] = {}
        for server_id, weight in self.workload.allocation_weights.items():
            try:
                rack_id = self._require_server(server_id).rack_id
            except SimulationError:
                continue
            weights_by_rack[rack_id] = weights_by_rack.get(rack_id, 0.0) + weight
        total_weight = sum(weights_by_rack.values())
        self.workload.workload_placement_imbalance_ratio = round(
            max(weights_by_rack.values(), default=0.0) / max(total_weight, 0.001),
            4,
        )
        forbidden_rack_ids = set(self.workload.forbidden_rack_ids or [])
        violating_racks = {
            rack_id
            for rack_id, weight in weights_by_rack.items()
            if weight > 0.0 and rack_id in forbidden_rack_ids
        }
        if self.workload.placement_policy_status == "misconfigured" and self.workload.placement_policy_target_rack_id:
            if weights_by_rack.get(self.workload.placement_policy_target_rack_id, 0.0) > 0.0:
                violating_racks.add(self.workload.placement_policy_target_rack_id)
        self.workload.placement_policy_violating_racks = len(violating_racks)

    def _thermal_throttle_effective_severity(self, fault: Fault) -> float:
        severity = fault.severity
        strong_cooling = any(
            control.action_type == "set_cooling"
            and control.details.get("target")
            and self.cooling_units
            and any(
                unit.cooling_unit_id == control.details.get("target")
                and unit.fan_speed_percent >= 95.0
                and unit.supply_air_temperature_c <= 17.0
                for unit in self.cooling_units
            )
            for control in self.controls
        )
        if strong_cooling:
            return 0.0
        if self.workload.throttle_rate_per_second is not None and self.workload.throttle_rate_per_second <= 500:
            return 0.0
        if not self._workload_allocated_to_target(fault.target):
            return 0.0
        return severity

    def _recalculate(self) -> None:
        rack_fault_offsets: dict[str, float] = {}
        power_overload_targets: dict[str, float] = {}
        power_budget_targets: dict[str, float] = {}
        power_budget_limits: dict[str, float] = {}
        failed_servers: set[str] = set()
        intermittent_failed_servers: set[str] = set()
        sensor_bias_by_target: dict[str, tuple[float, str]] = {}
        thermal_throttle_targets: dict[str, float] = {}
        tor_packet_loss_targets: dict[str, float] = {}
        autoscaler_misconfiguration_severity = 0.0
        autoscaler_policy_updated = False
        monitoring_pipeline_failure_severity = 0.0
        monitoring_pipeline_repaired = False
        placement_policy_target: str | None = None
        placement_policy_severity = 0.0
        placement_policy_updated = False
        load_balancer_misconfiguration_severity = 0.0
        load_balancer_config_updated = False
        degraded_cooling: dict[str, float] = {}
        global_cooling_offset = 0.0

        for fault in self.active_faults.values():
            if fault.fault_type == "cooling_degradation":
                degraded_cooling[fault.target] = max(degraded_cooling.get(fault.target, 0.0), fault.severity)
                global_cooling_offset += fault.severity * 12.0
            elif fault.fault_type == "rack_hotspot":
                rack_fault_offsets[fault.target] = rack_fault_offsets.get(fault.target, 0.0) + fault.severity * 15.0
            elif fault.fault_type == "power_overload":
                power_overload_targets[fault.target] = max(power_overload_targets.get(fault.target, 0.0), fault.severity)
            elif fault.fault_type == "server_failure":
                failed_servers.add(fault.target)
            elif fault.fault_type == "tor_packet_loss":
                tor_packet_loss_targets[fault.target] = max(
                    tor_packet_loss_targets.get(fault.target, 0.0),
                    fault.severity,
                )
            elif fault.fault_type == "autoscaler_misconfiguration":
                if self._control_applied_after_fault_started("update_autoscaler_policy", fault):
                    autoscaler_policy_updated = True
                else:
                    autoscaler_misconfiguration_severity = max(
                        autoscaler_misconfiguration_severity,
                        fault.severity,
                    )
            elif fault.fault_type == "monitoring_pipeline_failure":
                if self._monitoring_pipeline_repaired_for_fault(fault):
                    monitoring_pipeline_repaired = True
                else:
                    monitoring_pipeline_failure_severity = max(
                        monitoring_pipeline_failure_severity,
                        fault.severity,
                    )
            elif fault.fault_type == "placement_policy_misconfiguration":
                if self._control_applied_after_fault_started("update_placement_policy", fault):
                    placement_policy_updated = True
                elif fault.severity >= placement_policy_severity:
                    placement_policy_target = fault.target
                    placement_policy_severity = fault.severity
            elif fault.fault_type == "load_balancer_misconfiguration":
                if self._control_applied_after_fault_started("update_load_balancer_config", fault):
                    load_balancer_config_updated = True
                else:
                    load_balancer_misconfiguration_severity = max(
                        load_balancer_misconfiguration_severity,
                        fault.severity,
                    )
            elif fault.fault_type == "thermal_sensor_miscalibration":
                raw_bias_c = fault.severity * 14.0
                sensor_bias_by_target[fault.target] = self._calibrated_sensor_bias(fault.target, raw_bias_c)
            elif fault.fault_type == "power_budget_violation":
                power_budget_targets[fault.target] = max(power_budget_targets.get(fault.target, 0.0), fault.severity)
                injected_budget = fault.parameters.get(
                    "injected_power_budget_kw"
                )
                if (
                    isinstance(injected_budget, (int, float))
                    and not isinstance(injected_budget, bool)
                    and injected_budget > 0.0
                ):
                    power_budget_limits[fault.target] = min(
                        power_budget_limits.get(fault.target, float("inf")),
                        float(injected_budget),
                    )
            elif fault.fault_type == "intermittent_server_failure" and self._intermittent_failure_is_active(fault):
                intermittent_failed_servers.add(fault.target)
            elif fault.fault_type == "thermal_throttling":
                effective_severity = self._thermal_throttle_effective_severity(fault)
                if effective_severity > 0.0:
                    thermal_throttle_targets[fault.target] = max(
                        thermal_throttle_targets.get(fault.target, 0.0),
                        effective_severity,
                    )
                    rack_id = self._require_server(fault.target).rack_id if fault.target.startswith("server-") else fault.target
                    rack_fault_offsets[rack_id] = rack_fault_offsets.get(rack_id, 0.0) + effective_severity * 10.0

        # Recalculate cooling unit capacity based on degradation and control settings
        for unit in self.cooling_units:
            severity = degraded_cooling.get(unit.cooling_unit_id, 0.0)
            fan_factor = 0.75 + (unit.fan_speed_percent / 100.0) * 0.5
            temp_factor = max(0.6, 1.0 + (20.0 - unit.supply_air_temperature_c) * 0.03)
            unit.cooling_capacity_kw = unit.baseline_capacity_kw * (1.0 - severity) * fan_factor * temp_factor
            unit.status = "degraded" if severity > 0 else "healthy"

        # Mitigation: cooling degradation can be partially offset by increasing fan speed 
        # or lowering supply air temperature, so apply a global offset to rack temperatures that can be reduced by controls
        if global_cooling_offset:
            avg_fan = sum(unit.fan_speed_percent for unit in self.cooling_units) / max(len(self.cooling_units), 1)
            avg_supply = sum(unit.supply_air_temperature_c for unit in self.cooling_units) / max(len(self.cooling_units), 1)
            mitigation_bonus = max(0.0, avg_fan - self.config.cooling.fan_speed_percent) * 0.1
            mitigation_bonus += max(0.0, self.config.cooling.supply_air_temperature_c - avg_supply) * 0.8
            global_cooling_offset = max(0.0, global_cooling_offset - mitigation_bonus)

        # Update server statuses and reset workload related metrics before applying new workload placement
        for server in self.servers:
            previous_health_check_status = server.health_check_status
            server.thermal_throttle_factor = 1.0
            server.cpu_frequency_scale = 1.0
            server.temperature_sensor_status = "normal"
            server.temperature_sensor_bias_c = 0.0
            server.temperature_sensor_disagreement_c = 0.0
            if server.server_id in failed_servers:
                server.status = "failed"
                server.health_check_status = "failing"
            elif server.server_id in intermittent_failed_servers and server.status != "maintenance":
                server.status = "failed"
                server.health_check_status = "flapping"
            elif server.status == "failed" and server.server_id not in failed_servers:
                server.status = "healthy"
            if server.status == "maintenance":
                server.health_check_status = "passing"
            if server.status not in {"maintenance", "failed"}:
                server.status = "healthy"
                server.health_check_status = "passing"
            if previous_health_check_status != server.health_check_status:
                server.health_status_change_count += 1
                server.last_health_status_change_sim_time_seconds = self.sim_time_seconds
                self._record_event(
                    "host_health_status_changed",
                    f"Host health status changed in {server.rack_id}",
                    {
                        "rack_id": server.rack_id,
                        "status": server.health_check_status,
                        "health_status_change_count": server.health_status_change_count,
                    },
                )
            for target, severity in thermal_throttle_targets.items():
                if target == server.server_id or target == server.rack_id:
                    server.thermal_throttle_factor = min(
                        server.thermal_throttle_factor,
                        max(0.2, 1.0 - severity * 0.75),
                    )
                    server.cpu_frequency_scale = server.thermal_throttle_factor
            server.cpu_utilization_percent = 0.0
            server.memory_utilization_percent = 0.0
            server.gpu_utilization_percent = 0.0
            server.workload_assigned = 0.0

        self._apply_autoscaler_state(autoscaler_misconfiguration_severity, autoscaler_policy_updated)
        self._apply_monitoring_pipeline_state(monitoring_pipeline_failure_severity, monitoring_pipeline_repaired)
        self._apply_placement_policy_state(placement_policy_target, placement_policy_severity, placement_policy_updated)
        self._apply_load_balancer_state(load_balancer_misconfiguration_severity, load_balancer_config_updated)

        self._expire_workload_jobs()
        self._place_thermally_limited_workload(
            power_overload_targets,
            power_budget_targets,
            power_budget_limits,
            sensor_bias_by_target,
            tor_packet_loss_targets,
            global_cooling_offset,
            rack_fault_offsets,
        )
        self._update_placement_policy_runtime_metrics()
        self._update_control_plane_operational_metrics()

        critical_racks = sum(rack.thermal_status == "critical" for rack in self.racks)
        overloaded_racks = sum(rack.power_status == "overloaded" for rack in self.racks)
        budget_violating_racks = sum(rack.power_budget_status == "violated" for rack in self.racks)
        failed_ratio = sum(server.status == "failed" for server in self.servers) / max(len(self.servers), 1)
        self.sla_status = (
            "violated"
            if critical_racks > 0
            or overloaded_racks > 0
            or budget_violating_racks > 0
            or failed_ratio > self.config.thresholds.failed_server_ratio_sla_threshold
            or self.workload.queue_length > self.config.thresholds.workload_queue_sla_threshold
            or self.workload.network_error_rate > 0.0
            or self.workload.application_error_rate_percent > 0.0
            else "normal"
        )
        if self.sla_status != self._last_sla_status:
            event = "sla_violation_started" if self.sla_status == "violated" else "sla_violation_resolved"
            self._record_event(event, f"SLA status {self.sla_status}")
            self._last_sla_status = self.sla_status
        self.healthy_server_count = sum(server.status == "healthy" for server in self.servers)

    def _temperature_throttle_factor(self, inlet_temperature_c: float) -> float:
        """Derate physical compute capacity above the rack warning threshold.

        Capacity is full at the warning threshold, half at critical, and
        bounded below by 20%. Reported (possibly biased) sensor values affect
        alerts/SLA only; they do not change the physical capacity limit.
        """
        warning = self.config.thresholds.rack_warning_temp_c
        critical = self.config.thresholds.rack_critical_temp_c
        thermal_pressure = max(0.0, (inlet_temperature_c - warning) / (critical - warning))
        return max(0.2, 1.0 - 0.5 * thermal_pressure)

    def _place_thermally_limited_workload(
        self,
        power_overload_targets: dict[str, float],
        power_budget_targets: dict[str, float],
        power_budget_limits: dict[str, float],
        sensor_bias_by_target: dict[str, tuple[float, str]],
        tor_packet_loss_targets: dict[str, float],
        global_cooling_offset: float,
        rack_fault_offsets: dict[str, float],
    ) -> None:
        # Placement determines power/temperature, which in turn limits service
        # capacity. Resolve this feedback within the same recalculation. Trial
        # placements must not consume extra time, queue demand, or RNG samples.
        workload_snapshot = self._snapshot_workload_state()
        rng_state = self.rng.getstate()
        explicit_factors = {server.server_id: server.thermal_throttle_factor for server in self.servers}
        for iteration in range(32):
            if iteration:
                self._restore_workload_state(deepcopy(workload_snapshot))
                self.rng.setstate(rng_state)
            for server in self.servers:
                server.cpu_utilization_percent = 0.0
                server.memory_utilization_percent = 0.0
                server.gpu_utilization_percent = 0.0
                server.workload_assigned = 0.0
            self._place_workload()
            self._update_power_and_temperatures(
                power_overload_targets,
                power_budget_targets,
                power_budget_limits,
                sensor_bias_by_target,
                tor_packet_loss_targets,
                global_cooling_offset,
                rack_fault_offsets,
            )
            next_factors = {}
            for rack in self.racks:
                physical_factor = self._temperature_throttle_factor(rack.inlet_temperature_c)
                for server in rack.servers:
                    next_factors[server.server_id] = min(
                        explicit_factors[server.server_id],
                        physical_factor if server.status == "healthy" else 1.0,
                    )
            max_change = max(
                (abs(next_factors[server.server_id] - server.thermal_throttle_factor) for server in self.servers),
                default=0.0,
            )
            if max_change < 1e-7 or iteration == 31:
                break
            for server in self.servers:
                target_factor = next_factors[server.server_id]
                # Damping after the initial update stabilizes the thermal/power
                # feedback near thresholds. Keep the last applied factors at
                # the iteration bound so capacity and telemetry stay consistent.
                server.thermal_throttle_factor = (
                    target_factor if iteration == 0 else (server.thermal_throttle_factor + target_factor) / 2.0
                )
                server.cpu_frequency_scale = server.thermal_throttle_factor

    def _update_power_and_temperatures(
        self,
        power_overload_targets: dict[str, float],
        power_budget_targets: dict[str, float],
        power_budget_limits: dict[str, float],
        sensor_bias_by_target: dict[str, tuple[float, str]],
        tor_packet_loss_targets: dict[str, float],
        global_cooling_offset: float,
        rack_fault_offsets: dict[str, float],
    ) -> None:
        # Compute power from the currently placed workload.
        total_it_power = 0.0
        available_cooling = sum(unit.cooling_capacity_kw for unit in self.cooling_units)

        for rack in self.racks:
            #power overload effect
            if rack.rack_id in power_overload_targets:
                for server in rack.servers:
                    if server.status == "healthy":
                        server.cpu_utilization_percent = min(100.0, server.cpu_utilization_percent + 55.0)
            #failed servers consume no power, maintenance servers consume reduced power, 
            #and healthy servers consume power based on utilization plus potential overload penalty
            for server in rack.servers:
                if server.status == "failed":
                    server.power_kw = 0.0
                elif server.status == "maintenance":
                    server.power_kw = self.config.server.idle_power_kw * 0.5
                else:
                    server.power_kw = self.config.server.idle_power_kw + (
                        server.cpu_utilization_percent / 100.0 * self.config.server.dynamic_power_kw
                    )
                    server.power_kw += (
                        server.gpu_utilization_percent / 100.0 * self.config.server.dynamic_power_kw * 0.8
                    )
                    if rack.rack_id in power_overload_targets:
                        server.power_kw += (
                            self.config.rack.power_limit_kw
                            * (1.1 + power_overload_targets[rack.rack_id])
                            / max(len(rack.servers), 1)
                        )
                total_it_power += server.power_kw

        cooling_surplus = max(0.0, available_cooling - total_it_power)

        # Recalculate rack-level state
        for rack in self.racks:
            rack.total_power_kw = sum(server.power_kw for server in rack.servers)
            rack.power_budget_kw = self.config.rack.power_budget_kw
            if rack.rack_id in power_budget_targets:
                severity = power_budget_targets[rack.rack_id]
                # Freeze the injected operating budget at fault onset.  This
                # makes the violation observable on small/lightly loaded
                # topologies while still allowing workload migration to move
                # draw below the fixed limit.
                rack.power_budget_kw = power_budget_limits.get(
                    rack.rack_id,
                    round(
                        self.config.rack.power_budget_kw
                        * max(0.05, 1.0 - severity),
                        4,
                    ),
                )
            rack.temperature_sensor_status = "normal"
            rack.temperature_sensor_bias_c = 0.0
            rack.temperature_sensor_disagreement_c = 0.0
            rack.network_packet_loss_percent = 0.0
            rack.network_retransmit_rate = 0.0
            rack.network_error_rate = 0.0
            rack.network_path_status = "normal"
            rack.thermal_throttle_factor = min((server.thermal_throttle_factor for server in rack.servers), default=1.0)
            rack.average_cpu_utilization_percent = sum(server.cpu_utilization_percent for server in rack.servers) / max(
                len(rack.servers), 1
            )
            rack.inlet_temperature_c = (
                self.config.ambient_temperature_c
                + self.config.rack.heat_gain_factor * rack.total_power_kw
                - self.config.rack.cooling_effect_factor * cooling_surplus
                + global_cooling_offset
                + rack_fault_offsets.get(rack.rack_id, 0.0)
            )
            rack.outlet_temperature_c = rack.inlet_temperature_c + rack.total_power_kw * 0.35
            rack.reported_inlet_temperature_c = rack.inlet_temperature_c
            rack.reported_outlet_temperature_c = rack.outlet_temperature_c
            sensor_bias = sensor_bias_by_target.get(rack.rack_id)
            if sensor_bias is not None:
                bias_c, status = sensor_bias
                rack.temperature_sensor_status = status
                rack.temperature_sensor_bias_c = round(bias_c, 4)
                rack.reported_inlet_temperature_c = rack.inlet_temperature_c + bias_c
                rack.reported_outlet_temperature_c = rack.outlet_temperature_c + bias_c
                rack.temperature_sensor_disagreement_c = round(
                    abs(rack.reported_inlet_temperature_c - rack.inlet_temperature_c),
                    4,
                )
            thermal_inlet_c = rack.reported_inlet_temperature_c
            if thermal_inlet_c >= self.config.thresholds.rack_critical_temp_c:
                rack.thermal_status = "critical"
            elif thermal_inlet_c >= self.config.thresholds.rack_warning_temp_c:
                rack.thermal_status = "warning"
            else:
                rack.thermal_status = "normal"
            rack.power_status = "overloaded" if rack.total_power_kw > self.config.rack.power_limit_kw else "normal"
            rack.power_budget_status = "violated" if rack.total_power_kw > rack.power_budget_kw else "normal"
            if rack.rack_id in tor_packet_loss_targets:
                severity = tor_packet_loss_targets[rack.rack_id]
                rack.network_packet_loss_percent = round(severity * 12.0, 4)
                rack.network_retransmit_rate = round(severity * 28.0, 4)
                rack.network_error_rate = round(severity * 6.0, 4)
                rack.network_path_status = "degraded"
            for server in rack.servers:
                server.temperature_c = rack.outlet_temperature_c
                server.reported_temperature_c = server.temperature_c
                sensor_bias = sensor_bias_by_target.get(server.server_id)
                if sensor_bias is not None:
                    bias_c, status = sensor_bias
                    server.temperature_sensor_status = status
                    server.temperature_sensor_bias_c = round(bias_c, 4)
                    server.reported_temperature_c = server.temperature_c + bias_c
                    server.temperature_sensor_disagreement_c = round(
                        abs(server.reported_temperature_c - server.temperature_c),
                        4,
                    )
        
        #Facility-level power and PUE
        cooling_load = min(total_it_power, available_cooling)
        self.total_it_power_kw = round(total_it_power, 4)
        self.total_cooling_power_kw = round(cooling_load / max(self.config.cooling.cooling_efficiency, 0.001), 4)
        self.facility_power_kw = round(self.total_it_power_kw + self.total_cooling_power_kw, 4)
        self.pue = round(self.facility_power_kw / max(self.total_it_power_kw, 0.001), 4)

    def _update_control_plane_operational_metrics(self) -> None:
        """Model source-observable scheduler/API health.

        These are operational measurements, not fault labels.  The simulator
        dynamics may use the active incident to produce the physical effect,
        but the public telemetry adapter only reads the resulting latency and
        pending-operation gauges.
        """
        severity = self._max_fault_severity("control_plane_degradation")
        demand = (
            self.workload.current_demand_per_second
            if self.workload.running
            else 0.0
        )
        configured = max(self.workload.request_rate_per_second, 1.0)
        demand_pressure = min(2.0, demand / configured)
        queue_pressure = min(50.0, self.workload.queue_length / configured)
        self.workload.scheduler_api_latency_ms = round(
            2.0
            + 2.0 * demand_pressure
            + queue_pressure
            + 80.0 * severity,
            4,
        )
        self.workload.scheduler_pending_operations = max(
            0,
            int(
                round(
                    self.workload.queue_length * 0.02
                    + demand_pressure * severity * 25.0
                )
            ),
        )

    # place workload into an appropriate server and update the related simulator state.
    def _place_workload(self) -> None:
        if self._multi_tenant_enabled:
            self._place_tenant_workloads()
            return
        if not self.workload.running:
            self.workload.queue_length = 0
            self._last_queue_update_sim_time = self.sim_time_seconds
            self.workload.current_demand_per_second = 0.0
            self.workload.uncapped_demand_per_second = 0.0
            self.workload.current_profile_type = self.workload.workload_profile_type
            self.workload.maintenance_window_active = False
            self.workload.maintenance_affected_server_ids = []
            self.workload.maintenance_affected_server_workload_fraction = 1.0
            self._clear_workload_allocation()
            self._update_workload_impact([], 0.0, 0.0)
            return
        generated_demand = self._generate_workload_demand()
        demand_rate = generated_demand.request_rate_per_second
        healthy_servers = [server for server in self.servers if server.status == "healthy"]
        elapsed_seconds = max(0, self.sim_time_seconds - self._last_queue_update_sim_time)
    
        if not healthy_servers:
            self._set_workload_effective_allocation_weights({})
            if elapsed_seconds:
                self.workload.queue_length += int(demand_rate * elapsed_seconds)
                self._last_queue_update_sim_time = self.sim_time_seconds
            self._update_workload_impact([], 0.0, demand_rate)
            return

        if not self.workload.desired_allocation_weights:
            self._initialize_workload_allocation(healthy_servers)
        recipient_weights = self._effective_workload_allocation_weights(healthy_servers, generated_demand)
        recipients = [server for server in healthy_servers if recipient_weights.get(server.server_id, 0.0) > 0.0]
        if not recipients:
            if elapsed_seconds:
                self.workload.queue_length += int(demand_rate * elapsed_seconds)
                self._last_queue_update_sim_time = self.sim_time_seconds
            self._update_workload_impact([], 0.0, demand_rate)
            return
        recipients = [server for server in recipients if recipient_weights.get(server.server_id, 0.0) > 0.0]
        service_capacity = self._service_capacity(recipients, recipient_weights)
        previous_queue_length = self.workload.queue_length
        if elapsed_seconds:
            incoming = demand_rate * elapsed_seconds
            processed = service_capacity * elapsed_seconds
            self.workload.queue_length = max(0, int(round(previous_queue_length + incoming - processed)))
            self._last_queue_update_sim_time = self.sim_time_seconds

        queued_demand_rate = previous_queue_length / elapsed_seconds if elapsed_seconds else 0.0
        processed_rate = min(service_capacity, demand_rate + queued_demand_rate)
        total_cpu = processed_rate * self._effective_cpu_cost_per_request() * 100.0
        total_memory = (
            processed_rate
            * self.workload.memory_cost_per_request_mb
            * self._resource_demand_scale("memory")
            / 1024.0
        )
        total_weight = sum(recipient_weights.get(server.server_id, 0.0) for server in recipients)
        for server in recipients:
            jitter = 1.0
            if self.workload.noise_enabled:
                jitter += self.rng.gauss(0.0, self.workload.noise_stddev)
            share = recipient_weights.get(server.server_id, 0.0) / max(total_weight, 0.001)
            assigned_cpu = max(0.0, total_cpu * share * jitter)
            server.cpu_utilization_percent = min(100.0, assigned_cpu)
            server.memory_utilization_percent = min(100.0, total_memory * share)
            gpu_intensity = self._workload_class.gpu_demand / max(self._workload_class.cpu_demand, 0.001)
            server.gpu_utilization_percent = min(100.0, server.cpu_utilization_percent * gpu_intensity)
            server.workload_assigned = server.cpu_utilization_percent / max(
                self._effective_cpu_cost_per_request() * 100.0,
                0.001,
            )
        self._update_workload_impact(recipients, service_capacity, demand_rate)

    def _initialize_workload_allocation(self, healthy_servers: list[Server]) -> None:
        self._set_workload_desired_allocation_weights(self._candidate_workload_allocation_weights(healthy_servers))

    def _candidate_workload_allocation_weights(self, healthy_servers: list[Server]) -> dict[str, float]:
        if self.workload.placement_strategy == "rack_hotspot" and self.workload.target_rack_id:
            target_servers = [server for server in healthy_servers if server.rack_id == self.workload.target_rack_id]
            recipients = target_servers or healthy_servers
        elif self.workload.placement_strategy == "random":
            recipients = healthy_servers[:]
            self.rng.shuffle(recipients)
            recipients = recipients[: max(1, len(recipients) // 2)]
        else:
            recipients = healthy_servers
        forbidden_rack_ids = set(self.workload.forbidden_rack_ids or [])
        if forbidden_rack_ids:
            allowed_recipients = [server for server in recipients if server.rack_id not in forbidden_rack_ids]
            if not allowed_recipients:
                allowed_recipients = [server for server in healthy_servers if server.rack_id not in forbidden_rack_ids]
            recipients = allowed_recipients or recipients
        if self.workload.max_server_count is not None:
            recipients = recipients[: max(0, self.workload.max_server_count)]
        return {server.server_id: 1.0 for server in recipients}

    def _normalized_server_weights(self, weights: dict[str, float]) -> dict[str, float]:
        normalized = {
            server_id: round(weight, 8)
            for server_id, weight in weights.items()
            if weight > 0.0 and self._server_exists(server_id)
        }
        return dict(sorted(normalized.items()))

    def _set_workload_desired_allocation_weights(self, weights: dict[str, float]) -> None:
        normalized = self._normalized_server_weights(weights)
        self.workload.desired_allocation_weights = normalized
        self.workload.desired_allocated_server_ids = list(normalized)

    def _set_workload_effective_allocation_weights(self, weights: dict[str, float]) -> None:
        normalized = self._normalized_server_weights(weights)
        self.workload.allocation_weights = dict(sorted(normalized.items()))
        self.workload.allocated_server_ids = list(self.workload.allocation_weights)

    def _effective_workload_allocation_weights(
        self,
        healthy_servers: list[Server],
        generated_demand: GeneratedWorkloadDemand,
    ) -> dict[str, float]:
        healthy_ids = {server.server_id for server in healthy_servers}
        desired_weights = self.workload.desired_allocation_weights or self.workload.allocation_weights
        weights = {
            server_id: weight
            for server_id, weight in desired_weights.items()
            if server_id in healthy_ids and weight > 0.0
        }
        if generated_demand.maintenance_active:
            weights = {
                server_id: (
                    weight * generated_demand.affected_server_workload_fraction
                    if server_id in generated_demand.affected_server_ids
                    else weight
                )
                for server_id, weight in weights.items()
            }
        weights = self._apply_autoscaler_limit_to_weights(weights)
        weights = self._apply_load_balancer_to_weights(weights)
        weights = self._apply_network_partition_to_weights(weights)
        self._set_workload_effective_allocation_weights(weights)
        return dict(self.workload.allocation_weights)

    def _apply_autoscaler_limit_to_weights(self, weights: dict[str, float]) -> dict[str, float]:
        limit = self.workload.autoscaler_effective_server_limit
        if limit is None or limit <= 0:
            return weights
        active_items = [(server_id, weight) for server_id, weight in sorted(weights.items()) if weight > 0.0]
        if len(active_items) <= limit:
            return weights
        allowed_server_ids = {server_id for server_id, _weight in active_items[:limit]}
        return {
            server_id: (weight if server_id in allowed_server_ids else 0.0)
            for server_id, weight in weights.items()
        }

    def _migrate_workload_allocation(self, source_rack_id: str, target_rack_id: str, workload_fraction: float) -> None:
        healthy_servers = [server for server in self.servers if server.status == "healthy"]
        if not healthy_servers:
            raise SimulationError("migrate_workload requires at least one healthy server")
        target_servers = [server for server in healthy_servers if server.rack_id == target_rack_id]
        if not target_servers:
            raise SimulationError(f"target rack has no healthy servers: {target_rack_id}")
        current_weights = (
            dict(self.workload.desired_allocation_weights)
            if self.workload.desired_allocation_weights
            else self._candidate_workload_allocation_weights(healthy_servers)
        )
        source_server_ids = {server.server_id for server in self.servers if server.rack_id == source_rack_id}
        source_weight = sum(
            weight
            for server_id, weight in current_weights.items()
            if server_id in source_server_ids
        )
        if source_weight <= 0.0:
            raise SimulationError(f"source rack has no allocated workload: {source_rack_id}")
        migrated_weight = source_weight * workload_fraction
        remaining_fraction = 1.0 - workload_fraction
        weights = dict(current_weights)
        for server_id in source_server_ids:
            if server_id in weights:
                weights[server_id] *= remaining_fraction
        per_target_server = migrated_weight / max(len(target_servers), 1)
        for server in target_servers:
            weights[server.server_id] = weights.get(server.server_id, 0.0) + per_target_server
        self._set_workload_desired_allocation_weights(weights)

    def _place_tenant_workloads(self) -> None:
        elapsed_seconds = max(0, self.sim_time_seconds - self._last_queue_update_sim_time)
        if not self.workload.running:
            self.workload.queue_length = 0
            self._last_queue_update_sim_time = self.sim_time_seconds
            for tenant in self.tenants.values():
                self._clear_tenant_allocation(tenant)
                tenant.queue_length = 0
                self._reset_tenant_runtime(tenant)
            self._update_workload_impact([], 0.0, 0.0)
            self._sync_tenant_state()
            return

        healthy_servers = [server for server in self.servers if server.status == "healthy"]
        remaining_cpu_by_server = {
            server.server_id: 100.0 * server.thermal_throttle_factor
            for server in healthy_servers
        }
        active_tenants = [tenant for tenant in self.tenants.values() if tenant.running]

        if not active_tenants:
            self.workload.current_demand_per_second = 0.0
            self.workload.uncapped_demand_per_second = 0.0
            self.workload.active_tenant_count = 0
            self.workload.service_capacity_requests_per_second = 0.0
            self.workload.queue_length = 0
            self._last_queue_update_sim_time = self.sim_time_seconds
            for tenant in self.tenants.values():
                self._clear_tenant_allocation(tenant)
                tenant.queue_length = 0
                self._reset_tenant_runtime(tenant)
            self._update_workload_impact([], 0.0, 0.0)
            self._sync_tenant_state()
            return

        generated_by_tenant: dict[str, GeneratedWorkloadDemand] = {}
        recipients_by_tenant: dict[str, list[Server]] = {}
        weights_by_tenant: dict[str, dict[str, float]] = {}
        previous_queue_by_tenant: dict[str, int] = {}

        for tenant in active_tenants:
            generated = self._generate_tenant_workload_demand(tenant)
            if not tenant.desired_allocation_weights:
                self._initialize_tenant_allocation(tenant, healthy_servers)
            recipients = self._tenant_recipients(tenant, healthy_servers)
            weights = self._tenant_recipient_weights(tenant, recipients, generated)
            recipients = [server for server in recipients if weights.get(server.server_id, 0.0) > 0.0]
            generated_by_tenant[tenant.tenant_id] = generated
            recipients_by_tenant[tenant.tenant_id] = recipients
            weights_by_tenant[tenant.tenant_id] = weights
            previous_queue_by_tenant[tenant.tenant_id] = tenant.queue_length

        ordered_tenants = sorted(active_tenants, key=lambda tenant: (-tenant.priority, tenant.tenant_id))
        for tenant in ordered_tenants:
            recipients = recipients_by_tenant[tenant.tenant_id]
            weights = weights_by_tenant[tenant.tenant_id]
            demand_rate = generated_by_tenant[tenant.tenant_id].request_rate_per_second
            previous_queue_length = previous_queue_by_tenant[tenant.tenant_id]
            if not recipients:
                if elapsed_seconds:
                    tenant.queue_length += int(demand_rate * elapsed_seconds)
                self._set_tenant_effective_allocation_weights(tenant, {})
                self._reset_tenant_runtime(tenant, demand_rate=demand_rate)
                self._update_tenant_demand_metrics(tenant, demand_rate)
                continue

            cpu_cost = self._effective_tenant_cpu_cost(tenant)
            available_cpu_percent = sum(
                remaining_cpu_by_server.get(server.server_id, 0.0) * weights.get(server.server_id, 0.0)
                for server in recipients
            )
            max_capacity = (
                available_cpu_percent
                / max(cpu_cost * 100.0, 0.001)
                * self._service_capacity_multiplier()
            )
            if tenant.quota_requests_per_second is not None:
                max_capacity = min(max_capacity, max(0.0, tenant.quota_requests_per_second))
            queued_demand_rate = previous_queue_length / elapsed_seconds if elapsed_seconds else 0.0
            processed_rate = min(max_capacity, demand_rate + queued_demand_rate)
            if elapsed_seconds:
                incoming = demand_rate * elapsed_seconds
                processed = processed_rate * elapsed_seconds
                tenant.queue_length = max(0, int(round(previous_queue_length + incoming - processed)))

            total_cpu = processed_rate * cpu_cost * 100.0
            allocation_weights = {
                server.server_id: remaining_cpu_by_server.get(server.server_id, 0.0) * weights.get(server.server_id, 0.0)
                for server in recipients
            }
            total_allocation_weight = sum(allocation_weights.values())
            served_allocation_weights: dict[str, float] = {}
            tenant_cpu_usage = 0.0
            tenant_memory_usage = 0.0
            tenant_gpu_usage = 0.0
            tenant_class = self._tenant_classes[tenant.tenant_id]
            for server in recipients:
                share = allocation_weights.get(server.server_id, 0.0) / max(total_allocation_weight, 0.001)
                assigned_cpu = min(remaining_cpu_by_server.get(server.server_id, 0.0), total_cpu * share)
                if assigned_cpu <= 0.0:
                    continue
                remaining_cpu_by_server[server.server_id] = max(
                    0.0,
                    remaining_cpu_by_server.get(server.server_id, 0.0) - assigned_cpu,
                )
                server.cpu_utilization_percent = min(100.0, server.cpu_utilization_percent + assigned_cpu)
                memory_usage = (
                    processed_rate
                    * self.workload.memory_cost_per_request_mb
                    * self._tenant_resource_demand_scale(tenant, "memory")
                    / 1024.0
                    * share
                )
                server.memory_utilization_percent = min(100.0, server.memory_utilization_percent + memory_usage)
                gpu_usage = assigned_cpu * tenant_class.gpu_demand / max(tenant_class.cpu_demand, 0.001)
                server.gpu_utilization_percent = min(100.0, server.gpu_utilization_percent + gpu_usage)
                server.workload_assigned += processed_rate * share
                served_allocation_weights[server.server_id] = allocation_weights.get(server.server_id, 0.0)
                tenant_cpu_usage += assigned_cpu
                tenant_memory_usage += memory_usage
                tenant_gpu_usage += gpu_usage

            self._set_tenant_effective_allocation_weights(tenant, served_allocation_weights)
            tenant.service_capacity_requests_per_second = round(max_capacity, 4)
            tenant.processed_rate_per_second = round(processed_rate, 4)
            tenant.cpu_usage_percent = round(tenant_cpu_usage / max(len(recipients), 1), 4)
            tenant.memory_usage_percent = round(tenant_memory_usage / max(len(recipients), 1), 4)
            tenant.gpu_utilization_percent = round(tenant_gpu_usage / max(len(recipients), 1), 4)
            self._update_tenant_demand_metrics(tenant, demand_rate)

        if elapsed_seconds:
            self._last_queue_update_sim_time = self.sim_time_seconds

        total_network_demand = sum(tenant.network_demand_mbps for tenant in active_tenants)
        total_storage_demand = sum(tenant.storage_demand_iops for tenant in active_tenants)
        global_network_ratio = total_network_demand / max(self.workload.network_capacity_mbps, 0.001)
        global_storage_ratio = total_storage_demand / max(self._effective_storage_capacity_iops(), 0.001)

        for tenant in active_tenants:
            self._update_tenant_latency(
                tenant,
                recipients_by_tenant[tenant.tenant_id],
                generated_by_tenant[tenant.tenant_id].request_rate_per_second,
                global_network_ratio,
                global_storage_ratio,
            )
        for tenant in self.tenants.values():
            if not tenant.running:
                self._clear_tenant_allocation(tenant)
                tenant.queue_length = 0
                self._reset_tenant_runtime(tenant)

        self._update_multi_tenant_aggregate(active_tenants, total_network_demand, total_storage_demand)
        self._sync_tenant_state()

    def _generate_workload_demand(self) -> GeneratedWorkloadDemand:
        if self._workload_trace:
            return self._apply_workload_trace_record(self._workload_trace.record_at(self.sim_time_seconds))
        generated_demand = self._workload_profile.demand_at(self.sim_time_seconds)
        generated_demand = self._apply_workload_throttle(generated_demand)
        self.workload.current_demand_per_second = generated_demand.request_rate_per_second
        self.workload.current_profile_type = generated_demand.profile_type
        self.workload.maintenance_window_active = generated_demand.maintenance_active
        self.workload.maintenance_affected_server_ids = sorted(generated_demand.affected_server_ids)
        self.workload.maintenance_affected_server_workload_fraction = (
            generated_demand.affected_server_workload_fraction
        )
        return generated_demand

    def _generate_tenant_workload_demand(self, tenant: TenantWorkloadState) -> GeneratedWorkloadDemand:
        tenant_trace = self._tenant_traces.get(tenant.tenant_id)
        if tenant_trace:
            return self._apply_tenant_throttle(
                tenant,
                self._apply_tenant_trace_record(
                    tenant,
                    tenant_trace.record_at(self.sim_time_seconds),
                    tenant_trace,
                ),
            )
        if self._workload_trace:
            record = self._workload_trace.record_at(self.sim_time_seconds, tenant.tenant_id)
            if record is not None:
                return self._apply_tenant_throttle(
                    tenant,
                    self._apply_tenant_trace_record(tenant, record, self._workload_trace),
                )
        generated_demand = self._tenant_profiles[tenant.tenant_id].demand_at(self.sim_time_seconds)
        generated_demand = self._apply_tenant_throttle(tenant, generated_demand)
        tenant.current_demand_per_second = generated_demand.request_rate_per_second
        tenant.current_profile_type = generated_demand.profile_type
        tenant.maintenance_window_active = generated_demand.maintenance_active
        tenant.maintenance_affected_server_ids = sorted(generated_demand.affected_server_ids)
        tenant.maintenance_affected_server_workload_fraction = generated_demand.affected_server_workload_fraction
        return generated_demand

    def _apply_tenant_trace_record(
        self,
        tenant: TenantWorkloadState,
        record: TraceReplayRecord | None,
        trace: TraceReplayEngine,
    ) -> GeneratedWorkloadDemand:
        tenant.trace_replay_enabled = True
        tenant.trace_replay_progress = trace.progress(self.sim_time_seconds, record.tenant_id if record else None).to_dict()
        tenant.current_profile_type = "trace_replay"
        tenant.workload_profile_type = "trace_replay"
        if record is None:
            tenant.current_demand_per_second = 0.0
            return GeneratedWorkloadDemand(profile_type="trace_replay", request_rate_per_second=0.0)
        try:
            tenant_class = build_workload_class(record.workload_class)
        except ValueError as error:
            raise SimulationError(str(error)) from error
        self._tenant_classes[tenant.tenant_id] = tenant_class
        tenant.workload_class = tenant_class.workload_class
        tenant.active_workload_class = tenant_class.workload_class
        tenant.class_resource_demand = tenant_class.resource_demand()
        tenant.cpu_demand = tenant_class.cpu_demand
        tenant.memory_demand = tenant_class.memory_demand
        tenant.network_demand = tenant_class.network_demand
        tenant.storage_demand = tenant_class.storage_demand
        tenant.gpu_demand = tenant_class.gpu_demand
        for resource_name, value in record.resource_overrides().items():
            setattr(tenant, f"{resource_name}_demand", value)
        tenant.class_resource_demand = {
            "cpu": tenant.cpu_demand,
            "memory": tenant.memory_demand,
            "network": tenant.network_demand,
            "storage": tenant.storage_demand,
            "gpu": tenant.gpu_demand,
        }
        tenant.current_demand_per_second = record.request_rate_per_second
        return GeneratedWorkloadDemand(
            profile_type="trace_replay",
            request_rate_per_second=record.request_rate_per_second,
        )

    def _initialize_tenant_allocation(self, tenant: TenantWorkloadState, healthy_servers: list[Server]) -> None:
        self._set_tenant_desired_allocation_weights(
            tenant,
            self._candidate_tenant_allocation_weights(tenant, healthy_servers),
        )

    def _candidate_tenant_allocation_weights(
        self,
        tenant: TenantWorkloadState,
        healthy_servers: list[Server],
    ) -> dict[str, float]:
        if tenant.placement_strategy == "rack_hotspot" and tenant.target_rack_id:
            target_servers = [server for server in healthy_servers if server.rack_id == tenant.target_rack_id]
            recipients = target_servers or healthy_servers
        elif tenant.placement_strategy == "random":
            recipients = healthy_servers[:]
            tenant_rng = random.Random(f"{self.seed}:{tenant.tenant_id}:allocation")
            tenant_rng.shuffle(recipients)
            recipients = recipients[: max(1, len(recipients) // 2)]
        else:
            recipients = healthy_servers[:]
        if tenant.max_server_count is not None:
            recipients = recipients[: max(0, tenant.max_server_count)]
        return {server.server_id: 1.0 for server in recipients}

    def _set_tenant_desired_allocation_weights(
        self,
        tenant: TenantWorkloadState,
        weights: dict[str, float],
    ) -> None:
        normalized = self._normalized_server_weights(weights)
        tenant.desired_allocation_weights = normalized
        tenant.desired_allocated_server_ids = list(normalized)

    def _set_tenant_effective_allocation_weights(
        self,
        tenant: TenantWorkloadState,
        weights: dict[str, float],
    ) -> None:
        normalized = self._normalized_server_weights(weights)
        tenant.allocation_weights = normalized
        tenant.allocated_server_ids = list(normalized)

    def _tenant_recipients(self, tenant: TenantWorkloadState, healthy_servers: list[Server]) -> list[Server]:
        healthy_ids = {server.server_id for server in healthy_servers}
        recipients = [
            server
            for server in healthy_servers
            if tenant.desired_allocation_weights.get(server.server_id, 0.0) > 0.0 and server.server_id in healthy_ids
        ]
        return sorted(recipients, key=lambda server: server.server_id)

    def _tenant_recipient_weights(
        self,
        tenant: TenantWorkloadState,
        recipients: list[Server],
        generated_demand: GeneratedWorkloadDemand,
    ) -> dict[str, float]:
        recipient_ids = {server.server_id for server in recipients}
        weights = {
            server_id: weight
            for server_id, weight in tenant.desired_allocation_weights.items()
            if server_id in recipient_ids and weight > 0.0
        }
        if generated_demand.maintenance_active:
            weights = {
                server_id: (
                    weight * generated_demand.affected_server_workload_fraction
                    if server_id in generated_demand.affected_server_ids
                    else weight
                )
                for server_id, weight in weights.items()
            }
        weights = self._apply_network_partition_to_weights(weights)
        return self._normalized_server_weights(weights)

    def _migrate_tenant_allocation(
        self,
        tenant: TenantWorkloadState,
        source_rack_id: str,
        target_rack_id: str,
        workload_fraction: float,
    ) -> None:
        healthy_servers = [server for server in self.servers if server.status == "healthy"]
        if not healthy_servers:
            raise SimulationError("migrate_workload requires at least one healthy server")
        target_servers = [server for server in healthy_servers if server.rack_id == target_rack_id]
        if not target_servers:
            raise SimulationError(f"target rack has no healthy servers: {target_rack_id}")
        current_weights = (
            dict(tenant.desired_allocation_weights)
            if tenant.desired_allocation_weights
            else self._candidate_tenant_allocation_weights(tenant, healthy_servers)
        )
        source_server_ids = {server.server_id for server in self.servers if server.rack_id == source_rack_id}
        source_weight = sum(
            weight
            for server_id, weight in current_weights.items()
            if server_id in source_server_ids
        )
        if source_weight <= 0.0:
            raise SimulationError(f"source rack has no allocated workload for tenant {tenant.tenant_id}: {source_rack_id}")
        migrated_weight = source_weight * workload_fraction
        remaining_fraction = 1.0 - workload_fraction
        weights = dict(current_weights)
        for server_id in source_server_ids:
            if server_id in weights:
                weights[server_id] *= remaining_fraction
        per_target_server = migrated_weight / max(len(target_servers), 1)
        for server in target_servers:
            weights[server.server_id] = weights.get(server.server_id, 0.0) + per_target_server
        self._set_tenant_desired_allocation_weights(tenant, weights)

    def _reset_tenant_runtime(self, tenant: TenantWorkloadState, demand_rate: float = 0.0) -> None:
        tenant.current_demand_per_second = demand_rate if tenant.running else 0.0
        tenant.uncapped_demand_per_second = demand_rate if tenant.running else 0.0
        tenant.service_capacity_requests_per_second = 0.0
        tenant.processed_rate_per_second = 0.0
        tenant.average_latency_ms = 0.0
        tenant.p95_latency_ms = 0.0
        tenant.queueing_latency_ms = 0.0
        tenant.service_time_latency_ms = 0.0
        tenant.network_demand_mbps = 0.0
        tenant.network_congestion_ratio = 0.0
        tenant.network_latency_penalty_ms = 0.0
        tenant.network_packet_loss_percent = 0.0
        tenant.network_retransmit_rate = 0.0
        tenant.network_error_rate = 0.0
        tenant.affected_rack_id = None
        tenant.storage_demand_iops = 0.0
        tenant.storage_utilization_ratio = 0.0
        tenant.storage_latency_penalty_ms = 0.0
        tenant.application_error_rate_percent = 0.0
        tenant.dropped_requests_per_second = 0.0
        tenant.cpu_usage_percent = 0.0
        tenant.memory_usage_percent = 0.0
        tenant.gpu_utilization_percent = 0.0
        tenant.sla_status = "normal"

    def _update_tenant_demand_metrics(self, tenant: TenantWorkloadState, demand_rate: float) -> None:
        tenant.network_demand_mbps = round(
            demand_rate
            * self.workload.network_kb_per_request
            * self._tenant_resource_demand_scale(tenant, "network")
            * 8.0
            / 1024.0,
            4,
        )
        tenant.storage_demand_iops = round(
            demand_rate * self.workload.storage_io_per_request * self._tenant_resource_demand_scale(tenant, "storage"),
            4,
        )
        tenant.application_error_rate_percent = self._application_error_rate_percent(
            tenant.tenant_id,
            demand_rate,
        )
        tenant.dropped_requests_per_second = round(
            demand_rate * tenant.application_error_rate_percent / 100.0,
            4,
        )

    def _update_tenant_latency(
        self,
        tenant: TenantWorkloadState,
        recipients: list[Server],
        demand_rate: float,
        global_network_ratio: float,
        global_storage_ratio: float,
    ) -> None:
        tenant_class = self._tenant_classes[tenant.tenant_id]
        network_capacity_factor, burst_congestion_active, burst_affected_target = self._network_congestion_capacity_factor(
            tenant.allocation_weights,
            demand_rate,
            tenant,
        )
        tenant_network_ratio = max(
            global_network_ratio,
            tenant.network_demand_mbps / max(self.workload.network_capacity_mbps * network_capacity_factor, 0.001),
        )
        tenant.network_congestion_ratio = round(tenant_network_ratio, 4)
        tenant.storage_utilization_ratio = round(global_storage_ratio, 4)
        network_approach_penalty = self.workload.network_base_latency_ms * tenant_network_ratio
        network_overload_penalty = self.workload.network_congestion_penalty_ms * max(0.0, tenant_network_ratio - 0.7) / 0.3
        packet_loss_percent, retransmit_rate, network_error_rate, packet_loss_latency_ms, affected_rack_id = (
            self._tor_packet_loss_metrics_for_weights(tenant.allocation_weights)
        )
        tenant.network_packet_loss_percent = packet_loss_percent
        tenant.network_retransmit_rate = retransmit_rate
        tenant.network_error_rate = network_error_rate
        tenant.affected_rack_id = affected_rack_id or (
            burst_affected_target if burst_congestion_active and isinstance(burst_affected_target, str) else None
        )
        tenant.network_latency_penalty_ms = round(
            (
                network_approach_penalty
                + network_overload_penalty
                + packet_loss_latency_ms
            )
            * tenant_class.network_latency_sensitivity,
            4,
        )
        storage_approach_penalty = self.workload.storage_base_latency_ms * global_storage_ratio
        storage_overload_penalty = self.workload.storage_congestion_penalty_ms * max(0.0, global_storage_ratio - 0.75) / 0.25
        tenant.storage_latency_penalty_ms = round(
            (storage_approach_penalty + storage_overload_penalty) * tenant_class.storage_latency_sensitivity,
            4,
        )
        utilization_pressure = (
            min(0.99, demand_rate / max(tenant.service_capacity_requests_per_second, 0.001))
            if tenant.service_capacity_requests_per_second
            else 1.0
        )
        if recipients:
            utilization_pressure = max(utilization_pressure, min(0.99, tenant.cpu_usage_percent / 100.0))
        burst_pressure = max(0.0, demand_rate / max(tenant.request_rate_per_second, 0.001) - 1.0)
        utilization_pressure = min(0.99, utilization_pressure * (1.0 + burst_pressure * tenant_class.burst_sensitivity))
        tenant.service_time_latency_ms = round(
            self.workload.service_latency_ms
            * tenant_class.latency_sensitivity
            * (1.0 + utilization_pressure / max(0.05, 1.0 - utilization_pressure)),
            4,
        )
        if tenant.service_capacity_requests_per_second:
            queueing_latency_ms = tenant.queue_length / tenant.service_capacity_requests_per_second * 1000.0
        else:
            queueing_latency_ms = tenant.queue_length * 1.0
        tenant.queueing_latency_ms = round(queueing_latency_ms, 4)
        tenant.average_latency_ms = round(
            self.workload.base_latency_ms
            + tenant.service_time_latency_ms
            + tenant.queueing_latency_ms
            + tenant.network_latency_penalty_ms
            + tenant.storage_latency_penalty_ms
            + self._latency_fault_penalty(tenant.tenant_id),
            4,
        )
        tenant.p95_latency_ms = round(
            tenant.average_latency_ms
            + tenant.service_time_latency_ms * 0.5
            + tenant.queueing_latency_ms * 0.75
            + tenant.network_latency_penalty_ms * 0.75
            + tenant.storage_latency_penalty_ms * 0.5,
            4,
        )
        new_status = (
            "violated"
            if tenant.queue_length > self.config.thresholds.workload_queue_sla_threshold
            or tenant.network_error_rate > 0.0
            or tenant.application_error_rate_percent > 0.0
            or tenant.dropped_requests_per_second > 0.0
            else "normal"
        )
        if tenant.sla_status != "violated" and new_status == "violated":
            tenant.sla_violation_count += 1
        tenant.sla_status = new_status

    def _workload_recipient_weights(
        self,
        recipients: list[Server],
        generated_demand: GeneratedWorkloadDemand,
    ) -> dict[str, float]:
        if not generated_demand.maintenance_active:
            return {server.server_id: 1.0 for server in recipients}
        return {
            server.server_id: (
                generated_demand.affected_server_workload_fraction
                if server.server_id in generated_demand.affected_server_ids
                else 1.0
            )
            for server in recipients
        }

    def _service_capacity(self, recipients: list[Server], recipient_weights: dict[str, float] | None = None) -> float:
        cpu_cost = self._effective_cpu_cost_per_request()
        if not recipients or cpu_cost <= 0:
            return 0.0
        if recipient_weights is None:
            return (
                sum(server.thermal_throttle_factor for server in recipients)
                / cpu_cost
                * self._service_capacity_multiplier()
            )
        return (
            sum(
                recipient_weights.get(server.server_id, 0.0) * server.thermal_throttle_factor
                for server in recipients
            )
            / cpu_cost
            * self._service_capacity_multiplier()
        )

    def _effective_cpu_cost_per_request(self) -> float:
        return self.workload.cpu_cost_per_request * self._resource_demand_scale("cpu")

    def _resource_demand_scale(self, resource_name: str) -> float:
        current = getattr(self.workload, f"{resource_name}_demand")
        baseline = getattr(BASELINE_WORKLOAD_CLASS, f"{resource_name}_demand")
        return max(0.001, current / max(baseline, 0.001))

    def _effective_tenant_cpu_cost(self, tenant: TenantWorkloadState) -> float:
        return self.workload.cpu_cost_per_request * self._tenant_resource_demand_scale(tenant, "cpu")

    def _tenant_resource_demand_scale(self, tenant: TenantWorkloadState, resource_name: str) -> float:
        current = getattr(tenant, f"{resource_name}_demand")
        baseline = getattr(BASELINE_WORKLOAD_CLASS, f"{resource_name}_demand")
        return max(0.001, current / max(baseline, 0.001))

    def _update_multi_tenant_aggregate(
        self,
        active_tenants: list[TenantWorkloadState],
        total_network_demand: float,
        total_storage_demand: float,
    ) -> None:
        total_demand = sum(tenant.current_demand_per_second for tenant in active_tenants)
        total_uncapped_demand = sum(tenant.uncapped_demand_per_second for tenant in active_tenants)
        total_queue = sum(tenant.queue_length for tenant in active_tenants)
        total_dropped = sum(tenant.dropped_requests_per_second for tenant in active_tenants)
        self.workload.current_demand_per_second = round(total_demand, 4)
        self.workload.uncapped_demand_per_second = round(total_uncapped_demand, 4)
        self.workload.current_profile_type = "multi_tenant"
        self.workload.active_workload_class = "multi_tenant"
        self.workload.active_tenant_count = len(active_tenants)
        self.workload.queue_length = total_queue
        self.workload.service_capacity_requests_per_second = round(
            sum(tenant.service_capacity_requests_per_second for tenant in active_tenants),
            4,
        )
        self.workload.network_demand_mbps = round(total_network_demand, 4)
        self.workload.network_congestion_ratio = round(
            total_network_demand / max(self.workload.network_capacity_mbps, 0.001),
            4,
        )
        self.workload.storage_demand_iops = round(total_storage_demand, 4)
        self.workload.storage_utilization_ratio = round(
            total_storage_demand / max(self._effective_storage_capacity_iops(), 0.001),
            4,
        )
        self.workload.dropped_requests_per_second = round(total_dropped, 4)
        self.workload.application_error_rate_percent = round(
            total_dropped / total_demand * 100.0 if total_demand else 0.0,
            4,
        )
        self.workload.gpu_utilization_percent = round(
            sum(tenant.gpu_utilization_percent for tenant in active_tenants) / max(len(active_tenants), 1),
            4,
        )
        self.workload.cpu_demand = round(_weighted_average(active_tenants, "cpu_demand"), 4)
        self.workload.memory_demand = round(_weighted_average(active_tenants, "memory_demand"), 4)
        self.workload.network_demand = round(_weighted_average(active_tenants, "network_demand"), 4)
        self.workload.storage_demand = round(_weighted_average(active_tenants, "storage_demand"), 4)
        self.workload.gpu_demand = round(_weighted_average(active_tenants, "gpu_demand"), 4)
        self.workload.class_resource_demand = {
            "cpu": self.workload.cpu_demand,
            "memory": self.workload.memory_demand,
            "network": self.workload.network_demand,
            "storage": self.workload.storage_demand,
            "gpu": self.workload.gpu_demand,
        }
        self.workload.network_latency_penalty_ms = round(
            max((tenant.network_latency_penalty_ms for tenant in active_tenants), default=0.0),
            4,
        )
        self.workload.network_packet_loss_percent = round(
            max((tenant.network_packet_loss_percent for tenant in active_tenants), default=0.0),
            4,
        )
        self.workload.network_retransmit_rate = round(
            max((tenant.network_retransmit_rate for tenant in active_tenants), default=0.0),
            4,
        )
        self.workload.network_error_rate = round(
            max((tenant.network_error_rate for tenant in active_tenants), default=0.0),
            4,
        )
        self.workload.affected_rack_id = next(
            (
                tenant.affected_rack_id
                for tenant in active_tenants
                if tenant.affected_rack_id
            ),
            None,
        )
        self.workload.storage_latency_penalty_ms = round(
            max((tenant.storage_latency_penalty_ms for tenant in active_tenants), default=0.0),
            4,
        )
        if total_demand:
            self.workload.average_latency_ms = round(
                sum(tenant.average_latency_ms * tenant.current_demand_per_second for tenant in active_tenants)
                / total_demand,
                4,
            )
            self.workload.p95_latency_ms = round(max(tenant.p95_latency_ms for tenant in active_tenants), 4)
        else:
            self.workload.average_latency_ms = 0.0
            self.workload.p95_latency_ms = 0.0
        self.workload.queueing_latency_ms = round(
            max((tenant.queueing_latency_ms for tenant in active_tenants), default=0.0),
            4,
        )
        self.workload.service_time_latency_ms = round(
            max((tenant.service_time_latency_ms for tenant in active_tenants), default=0.0),
            4,
        )

    def _update_workload_impact(self, recipients: list[Server], service_capacity: float, demand_rate: float) -> None:
        if not self.workload.running:
            self.workload.service_capacity_requests_per_second = 0.0
            self.workload.network_demand_mbps = 0.0
            self.workload.network_congestion_ratio = 0.0
            self.workload.network_latency_penalty_ms = 0.0
            self.workload.network_packet_loss_percent = 0.0
            self.workload.network_retransmit_rate = 0.0
            self.workload.network_error_rate = 0.0
            self.workload.affected_rack_id = None
            self.workload.storage_demand_iops = 0.0
            self.workload.storage_utilization_ratio = 0.0
            self.workload.storage_latency_penalty_ms = 0.0
            self.workload.application_error_rate_percent = 0.0
            self.workload.dropped_requests_per_second = 0.0
            self.workload.load_balancer_request_skew_ratio = 0.0
            self.workload.load_balancer_unhealthy_routing_fraction = 0.0
            self.workload.load_balancer_error_rate_percent = 0.0
            self.workload.gpu_utilization_percent = 0.0
            self.workload.queueing_latency_ms = 0.0
            self.workload.service_time_latency_ms = 0.0
            self.workload.fault_latency_penalty_ms = 0.0
            self.workload.average_latency_ms = 0.0
            self.workload.p95_latency_ms = 0.0
            return

        self.workload.service_capacity_requests_per_second = round(service_capacity, 4)
        self.workload.network_demand_mbps = round(
            demand_rate
            * self.workload.network_kb_per_request
            * self._resource_demand_scale("network")
            * 8.0
            / 1024.0,
            4,
        )
        network_capacity_factor, burst_congestion_active, burst_affected_target = self._network_congestion_capacity_factor(
            self.workload.allocation_weights,
            demand_rate,
        )
        self.workload.network_congestion_ratio = round(
            self.workload.network_demand_mbps
            / max(self.workload.network_capacity_mbps * network_capacity_factor, 0.001),
            4,
        )

        congestion_ratio = self.workload.network_congestion_ratio
        approach_penalty = self.workload.network_base_latency_ms * congestion_ratio
        overload_penalty = self.workload.network_congestion_penalty_ms * max(0.0, congestion_ratio - 0.7) / 0.3
        packet_loss_percent, retransmit_rate, network_error_rate, packet_loss_latency_ms, affected_rack_id = (
            self._tor_packet_loss_metrics_for_weights(self.workload.allocation_weights)
        )
        self.workload.network_packet_loss_percent = packet_loss_percent
        self.workload.network_retransmit_rate = retransmit_rate
        self.workload.network_error_rate = network_error_rate
        self.workload.affected_rack_id = affected_rack_id or (
            burst_affected_target if burst_congestion_active and isinstance(burst_affected_target, str) else None
        )
        self.workload.network_latency_penalty_ms = round(
            (
                approach_penalty
                + overload_penalty
                + packet_loss_latency_ms
            )
            * self._workload_class.network_latency_sensitivity,
            4,
        )

        self.workload.storage_demand_iops = round(
            demand_rate * self.workload.storage_io_per_request * self._resource_demand_scale("storage"),
            4,
        )
        self.workload.storage_utilization_ratio = round(
            self.workload.storage_demand_iops / max(self._effective_storage_capacity_iops(), 0.001),
            4,
        )
        storage_ratio = self.workload.storage_utilization_ratio
        storage_approach_penalty = self.workload.storage_base_latency_ms * storage_ratio
        storage_overload_penalty = self.workload.storage_congestion_penalty_ms * max(0.0, storage_ratio - 0.75) / 0.25
        self.workload.storage_latency_penalty_ms = round(
            (storage_approach_penalty + storage_overload_penalty) * self._workload_class.storage_latency_sensitivity,
            4,
        )
        self._update_load_balancer_runtime_metrics()
        self.workload.application_error_rate_percent = max(
            self._application_error_rate_percent(demand_rate=demand_rate),
            self.workload.load_balancer_error_rate_percent,
        )
        self.workload.dropped_requests_per_second = round(
            demand_rate * self.workload.application_error_rate_percent / 100.0,
            4,
        )
        self.workload.gpu_utilization_percent = round(
            sum(server.gpu_utilization_percent for server in recipients) / max(len(recipients), 1) if recipients else 0.0,
            4,
        )

        utilization_pressure = min(0.99, demand_rate / max(service_capacity, 0.001)) if service_capacity else 1.0
        if recipients:
            recipient_utilization = sum(server.cpu_utilization_percent for server in recipients) / max(len(recipients), 1)
            utilization_pressure = max(utilization_pressure, min(0.99, recipient_utilization / 100.0))
        burst_pressure = max(0.0, demand_rate / max(self.workload.request_rate_per_second, 0.001) - 1.0)
        utilization_pressure = min(
            0.99,
            utilization_pressure * (1.0 + burst_pressure * self._workload_class.burst_sensitivity),
        )
        self.workload.service_time_latency_ms = round(
            self.workload.service_latency_ms
            * self._workload_class.latency_sensitivity
            * (1.0 + utilization_pressure / max(0.05, 1.0 - utilization_pressure)),
            4,
        )

        if service_capacity:
            queueing_latency_ms = self.workload.queue_length / service_capacity * 1000.0
        else:
            queueing_latency_ms = self.workload.queue_length * 1.0
        self.workload.queueing_latency_ms = round(queueing_latency_ms, 4)
        self.workload.fault_latency_penalty_ms = round(self._latency_fault_penalty(), 4)
        self.workload.average_latency_ms = round(
            self.workload.base_latency_ms
            + self.workload.service_time_latency_ms
            + self.workload.queueing_latency_ms
            + self.workload.network_latency_penalty_ms
            + self.workload.storage_latency_penalty_ms
            + self.workload.fault_latency_penalty_ms,
            4,
        )
        self.workload.p95_latency_ms = round(
            self.workload.average_latency_ms
            + self.workload.service_time_latency_ms * 0.5
            + self.workload.queueing_latency_ms * 0.75
            + self.workload.storage_latency_penalty_ms * 0.5
            + self.workload.network_latency_penalty_ms * 0.75,
            4,
        )

    def _latency_fault_penalty(self, tenant_id: str | None = None) -> float:
        penalty = 0.0
        for fault in self.active_faults.values():
            if fault.fault_type in {"power_overload", "server_failure"}:
                penalty += fault.severity * self.workload.latency_fault_penalty_ms
            elif fault.fault_type == "control_plane_degradation":
                penalty += fault.severity * self.workload.latency_fault_penalty_ms * 1.5
            elif fault.fault_type == "application_error" and self._application_fault_applies(fault, tenant_id):
                penalty += fault.severity * self.workload.latency_fault_penalty_ms * 0.5
        return penalty

    @property
    def racks(self) -> list[Rack]:
        return [rack for room in self.rooms for row in room.rows for rack in row.racks]

    @property
    def servers(self) -> list[Server]:
        return [server for rack in self.racks for server in rack.servers]

    def _require_rack(self, rack_id: str) -> Rack:
        for rack in self.racks:
            if rack.rack_id == rack_id:
                return rack
        raise SimulationError(f"rack not found: {rack_id}")

    def _require_server(self, server_id: str) -> Server:
        for server in self.servers:
            if server.server_id == server_id:
                return server
        raise SimulationError(f"server not found: {server_id}")

    def _require_tenant(self, tenant_id: str | None, action_name: str) -> TenantWorkloadState:
        if not tenant_id:
            raise SimulationError(f"{action_name} requires tenant_id")
        tenant = self.tenants.get(str(tenant_id))
        if tenant is None:
            raise SimulationError(f"tenant not found: {tenant_id}")
        return tenant

    def _server_exists(self, server_id: str) -> bool:
        return any(server.server_id == server_id for server in self.servers)

    def _require_cooling_unit(self, cooling_unit_id: str) -> CoolingUnit:
        for unit in self.cooling_units:
            if unit.cooling_unit_id == cooling_unit_id:
                return unit
        raise SimulationError(f"cooling unit not found: {cooling_unit_id}")

    def topology(self) -> dict[str, Any]:
        with self.lock:
            return {
                "rooms": [room.to_dict() for room in self.rooms],
                "cooling_units": [unit.to_dict() for unit in self.cooling_units],
            }

    def configuration_snapshot(self) -> dict[str, Any]:
        with self.lock:
            reset_config = self.config.model_dump(mode="json")
            current_config = self._current_configuration_snapshot()
            return {
                "config": reset_config,
                "reset_config": reset_config,
                "current": current_config,
                "simulation": current_config["simulation"],
                "topology": current_config["topology"],
                "thresholds": current_config["thresholds"],
                "workload": current_config["workload"],
                "tenants": current_config["tenants"],
                "cooling_units": current_config["cooling_units"],
                "active_controls": current_config["active_controls"],
                "supported_faults": sorted(SUPPORTED_FAULTS),
                "supported_actions": sorted(SUPPORTED_ACTIONS),
            }

    def alerts(self) -> list[dict[str, Any]]:
        with self.lock:
            alerts: list[dict[str, Any]] = []
            if self.sla_status == "violated":
                alerts.append(
                    _alert(
                        "sla_violation",
                        "critical",
                        "datacenter",
                        "Data center SLA is violated",
                        {
                            "thermal_critical": len([rack for rack in self.racks if rack.thermal_status == "critical"]),
                            "power_overloaded_racks": len([rack for rack in self.racks if rack.power_status == "overloaded"]),
                            "power_budget_violating_racks": len(
                                [rack for rack in self.racks if rack.power_budget_status == "violated"]
                            ),
                            "failed_servers": len([server for server in self.servers if server.status == "failed"]),
                            "workload_queue_length": self.workload.queue_length,
                            "network_error_rate": self.workload.network_error_rate,
                            "workload_application_error_rate_percent": self.workload.application_error_rate_percent,
                        },
                    )
                )
            for rack in self.racks:
                if rack.thermal_status in {"warning", "critical"}:
                    alerts.append(
                        _alert(
                            "rack_thermal",
                            rack.thermal_status,
                            rack.rack_id,
                            f"Rack {rack.rack_id} inlet temperature is {rack.thermal_status}",
                            {
                                "inlet_temperature_c": round(rack.inlet_temperature_c, 4),
                                "outlet_temperature_c": round(rack.outlet_temperature_c, 4),
                                "warning_threshold_c": self.config.thresholds.rack_warning_temp_c,
                                "critical_threshold_c": self.config.thresholds.rack_critical_temp_c,
                            },
                        )
                    )
                if rack.temperature_sensor_status != "normal":
                    alerts.append(
                        _alert(
                            "thermal_sensor_health",
                            "warning" if rack.thermal_status != "critical" else "critical",
                            rack.rack_id,
                            f"Temperature sensor readings are inconsistent for {rack.rack_id}",
                            {
                                "reported_inlet_temperature_c": round(rack.reported_inlet_temperature_c, 4),
                                "inlet_temperature_c": round(rack.inlet_temperature_c, 4),
                                "temperature_sensor_disagreement_c": rack.temperature_sensor_disagreement_c,
                                "temperature_sensor_status": rack.temperature_sensor_status,
                            },
                        )
                    )
                if rack.power_status == "overloaded":
                    alerts.append(
                        _alert(
                            "rack_power",
                            "critical",
                            rack.rack_id,
                            f"Rack {rack.rack_id} exceeds power limit",
                            {
                                "power_kw": round(rack.total_power_kw, 4),
                                "power_limit_kw": self.config.rack.power_limit_kw,
                            },
                        )
                    )
                if rack.power_budget_status == "violated":
                    alerts.append(
                        _alert(
                            "rack_power_budget",
                            "critical",
                            rack.rack_id,
                            f"Rack {rack.rack_id} exceeds configured power budget",
                            {
                                "power_kw": round(rack.total_power_kw, 4),
                                "power_budget_kw": rack.power_budget_kw,
                                "power_budget_status": rack.power_budget_status,
                            },
                        )
                    )
                if rack.thermal_throttle_factor < 0.999:
                    alerts.append(
                        _alert(
                            "rack_capacity_throttled",
                            "critical",
                            rack.rack_id,
                            f"Rack {rack.rack_id} service capacity is thermally reduced",
                            {
                                "thermal_throttle_factor": round(rack.thermal_throttle_factor, 4),
                                "reported_inlet_temperature_c": round(rack.reported_inlet_temperature_c, 4),
                            },
                        )
                    )
                if rack.network_packet_loss_percent > 0.0:
                    alerts.append(
                        _alert(
                            "PacketLossElevated",
                            "critical" if rack.network_packet_loss_percent >= 8.0 else "warning",
                            rack.rack_id,
                            f"Packet loss is elevated on rack path {rack.rack_id}",
                            {
                                "affected_rack_id": rack.rack_id,
                                "network_packet_loss_percent": rack.network_packet_loss_percent,
                                "network_retransmit_rate": rack.network_retransmit_rate,
                                "network_error_rate": rack.network_error_rate,
                            },
                        )
                    )
            for server in self.servers:
                if server.status == "failed":
                    alerts.append(
                        _alert(
                            "server_failed",
                            "critical",
                            server.server_id,
                            f"Server {server.server_id} is failed",
                            {"rack_id": server.rack_id},
                        )
                    )
                if server.health_check_status == "flapping":
                    alerts.append(
                        _alert(
                            "host_health_flapping",
                            "critical",
                            server.server_id,
                            f"Host health checks are flapping in {server.rack_id}",
                            {
                                "rack_id": server.rack_id,
                                "health_status_change_count": server.health_status_change_count,
                            },
                        )
                    )
                if server.temperature_sensor_status != "normal":
                    alerts.append(
                        _alert(
                            "host_temperature_sensor_health",
                            "warning",
                            server.server_id,
                            f"Host temperature sensor readings are inconsistent in {server.rack_id}",
                            {
                                "rack_id": server.rack_id,
                                "reported_temperature_c": round(server.reported_temperature_c, 4),
                                "temperature_c": round(server.temperature_c, 4),
                                "temperature_sensor_disagreement_c": server.temperature_sensor_disagreement_c,
                                "temperature_sensor_status": server.temperature_sensor_status,
                            },
                        )
                    )
                if server.thermal_throttle_factor < 0.999:
                    alerts.append(
                        _alert(
                            "host_capacity_throttled",
                            "critical",
                            server.server_id,
                            f"Host service capacity is thermally reduced in {server.rack_id}",
                            {
                                "rack_id": server.rack_id,
                                "thermal_throttle_factor": round(server.thermal_throttle_factor, 4),
                                "cpu_frequency_scale": round(server.cpu_frequency_scale, 4),
                            },
                        )
                    )
            if self.workload.queue_length > self.config.thresholds.workload_queue_sla_threshold:
                alerts.append(
                    _alert(
                        "workload_queue",
                        "critical",
                        self.workload.tenant_id if not self._multi_tenant_enabled else "multi_tenant",
                        "Workload queue exceeds SLA threshold",
                        {
                            "queue_length": self.workload.queue_length,
                            "threshold": self.config.thresholds.workload_queue_sla_threshold,
                        },
                    )
                )
            if self.workload.autoscaler_status == "misconfigured":
                alerts.append(
                    _alert(
                        "AutoscalerPolicyLimited",
                        "critical",
                        "autoscaler",
                        "Autoscaler policy is limiting effective service capacity",
                        {
                            "autoscaler_max_capacity": self.workload.autoscaler_max_capacity,
                            "autoscaler_target_utilization_percent": self.workload.autoscaler_target_utilization_percent,
                            "autoscaler_cooldown_seconds": self.workload.autoscaler_cooldown_seconds,
                            "autoscaler_effective_server_limit": self.workload.autoscaler_effective_server_limit,
                            "workload_service_capacity_requests_per_second": self.workload.service_capacity_requests_per_second,
                            "workload_current_demand_per_second": self.workload.current_demand_per_second,
                        },
                    )
                )
            if self.workload.telemetry_lag_seconds > 0 or self.workload.metrics_missing_ratio > 0.0:
                alerts.append(
                    _alert(
                        "TelemetryStale",
                        "critical" if self.workload.telemetry_lag_seconds >= 60 else "warning",
                        "monitoring-pipeline",
                        "Telemetry pipeline is stale or partially missing",
                        {
                            "metrics_last_updated_sim_time_seconds": self.workload.metrics_last_updated_sim_time_seconds,
                            "logs_last_updated_sim_time_seconds": self.workload.logs_last_updated_sim_time_seconds,
                            "telemetry_lag_seconds": self.workload.telemetry_lag_seconds,
                            "metrics_missing_ratio": self.workload.metrics_missing_ratio,
                            "logs_missing_ratio": self.workload.logs_missing_ratio,
                        },
                    )
                )
            if self.workload.placement_policy_violating_racks > 0:
                alerts.append(
                    _alert(
                        "PlacementPolicyDrift",
                        "critical",
                        self.workload.placement_policy_target_rack_id or "scheduler",
                        "Workload placement policy is concentrating traffic on a disallowed or risky rack",
                        {
                            "placement_policy_target_rack_id": self.workload.placement_policy_target_rack_id,
                            "placement_policy_violating_racks": self.workload.placement_policy_violating_racks,
                            "workload_placement_imbalance_ratio": self.workload.workload_placement_imbalance_ratio,
                        },
                    )
                )
            if self.workload.load_balancer_request_skew_ratio >= 0.35:
                alerts.append(
                    _alert(
                        "LoadBalancerBackendImbalance",
                        "critical" if self.workload.load_balancer_request_skew_ratio >= 0.6 else "warning",
                        "load-balancer",
                        "Load balancer backend request distribution is imbalanced",
                        {
                            "load_balancer_backend_skew_ratio": self.workload.load_balancer_request_skew_ratio,
                            "load_balancer_error_rate_percent": self.workload.load_balancer_error_rate_percent,
                            "workload_queue_length": self.workload.queue_length,
                            "workload_average_latency_ms": self.workload.average_latency_ms,
                        },
                    )
                )
            if self.workload.load_balancer_unhealthy_routing_fraction > 0.0:
                alerts.append(
                    _alert(
                        "LoadBalancerRoutingUnhealthyBackend",
                        "critical",
                        "load-balancer",
                        "Load balancer is routing requests to unhealthy backends",
                        {
                            "load_balancer_unhealthy_routing_fraction": (
                                self.workload.load_balancer_unhealthy_routing_fraction
                            ),
                            "load_balancer_error_rate_percent": self.workload.load_balancer_error_rate_percent,
                            "workload_application_error_rate_percent": self.workload.application_error_rate_percent,
                            "workload_dropped_requests_per_second": self.workload.dropped_requests_per_second,
                        },
                    )
                )
            if self.workload.network_congestion_ratio >= 0.85:
                alerts.append(
                    _alert(
                        "NetworkCongestionElevated",
                        "critical" if self.workload.network_congestion_ratio >= 1.0 else "warning",
                        self.workload.affected_rack_id or self.workload.tenant_id or "workload",
                        "Network congestion is elevated for the active workload",
                        {
                            "affected_rack_id": self.workload.affected_rack_id,
                            "workload_network_congestion_ratio": self.workload.network_congestion_ratio,
                            "workload_average_latency_ms": self.workload.average_latency_ms,
                            "workload_p95_latency_ms": self.workload.p95_latency_ms,
                            "workload_queue_length": self.workload.queue_length,
                        },
                    )
                )
            if self.workload.network_packet_loss_percent > 0.0:
                alerts.append(
                    _alert(
                        "PacketLossElevated",
                        "critical" if self.workload.network_packet_loss_percent >= 8.0 else "warning",
                        self.workload.affected_rack_id or self.workload.tenant_id or "workload",
                        "Packet loss and retransmits are elevated for the active workload",
                        {
                            "affected_rack_id": self.workload.affected_rack_id,
                            "network_packet_loss_percent": self.workload.network_packet_loss_percent,
                            "network_retransmit_rate": self.workload.network_retransmit_rate,
                            "network_error_rate": self.workload.network_error_rate,
                            "workload_average_latency_ms": self.workload.average_latency_ms,
                            "workload_queue_length": self.workload.queue_length,
                        },
                    )
                )
            if self.workload.application_error_rate_percent > 0.0:
                alerts.append(
                    _alert(
                        "application_errors",
                        "critical",
                        self.workload.tenant_id if not self._multi_tenant_enabled else "multi_tenant",
                        "Application errors are present",
                        {
                            "application_error_rate_percent": self.workload.application_error_rate_percent,
                            "dropped_requests_per_second": self.workload.dropped_requests_per_second,
                        },
                    )
                )
            for tenant in self.tenants.values():
                if tenant.sla_status == "violated":
                    alerts.append(
                        _alert(
                            "tenant_sla_violation",
                            "critical",
                            tenant.tenant_id,
                            f"Tenant {tenant.tenant_id} SLA is violated",
                            {
                                "queue_length": tenant.queue_length,
                                "application_error_rate_percent": tenant.application_error_rate_percent,
                                "dropped_requests_per_second": tenant.dropped_requests_per_second,
                                "sla_violation_count": tenant.sla_violation_count,
                            },
                        )
                    )
            for fault in self.active_faults.values():
                severity = "critical" if fault.severity >= 0.7 else "warning"
                alerts.append(
                    _alert(
                        "active_fault",
                        severity,
                        fault.target,
                        f"Active {fault.fault_type} fault on {fault.target}",
                        self._fault_telemetry(fault),
                    )
                )
            return alerts

    def observation(self, log_limit: int = 20, include_config: bool = True) -> dict[str, Any]:
        with self.lock:
            summary = self.state_summary()
            observation = {
                "sim_time_seconds": summary["sim_time_seconds"],
                "sla_status": summary["sla_status"],
                "summary": summary,
                "alerts": self.alerts(),
                "recent_events": self.recent_events(log_limit),
            }
            if include_config:
                observation["configuration"] = self._compact_configuration_snapshot()
            return observation

    def telemetry(self, log_limit: int = 50, include_config: bool = True) -> dict[str, Any]:
        with self.lock:
            summary = self.state_summary()
            telemetry = {
                "sim_time_seconds": summary["sim_time_seconds"],
                "summary": summary,
                "metrics": _structured_metrics(summary, self.racks, self.cooling_units, self.controls),
                "alerts": self.alerts(),
                "logs": self.recent_events(log_limit),
                "faults": [self._fault_telemetry(fault) for fault in self.active_faults.values()],
                "controls": self.list_controls(),
            }
            if include_config:
                telemetry["configuration"] = self.configuration_snapshot()
            return telemetry

    def _compact_configuration_snapshot(self) -> dict[str, Any]:
        return self.configuration_snapshot()

    def _current_configuration_snapshot(self) -> dict[str, Any]:
        return {
            "episode_id": self._episode_id,
            "seed": self.seed,
            "simulation": {
                **self.config.simulation.model_dump(mode="json"),
                "seed": self.seed,
            },
            "topology": self.config.topology.model_dump(mode="json"),
            "thresholds": self.config.thresholds.model_dump(mode="json"),
            "workload": {
                "running": self.workload.running,
                "tenant_id": self.workload.tenant_id,
                "request_rate_per_second": self.workload.request_rate_per_second,
                "configured_request_rate_per_second": self.workload.request_rate_per_second,
                "workload_class": self.workload.workload_class,
                "active_workload_class": self.workload.active_workload_class,
                "workload_profile_type": self.workload.workload_profile_type,
                "current_profile_type": self.workload.current_profile_type,
                "workload_profile_parameters": dict(self.workload.workload_profile_parameters),
                "placement_strategy": self.workload.placement_strategy,
                "target_rack_id": self.workload.target_rack_id,
                "forbidden_rack_ids": list(self.workload.forbidden_rack_ids),
                "max_server_count": self.workload.max_server_count,
                "placement_policy_status": self.workload.placement_policy_status,
                "autoscaler": {
                    "enabled": self.workload.autoscaler_enabled,
                    "min_capacity": self.workload.autoscaler_min_capacity,
                    "max_capacity": self.workload.autoscaler_max_capacity,
                    "target_utilization_percent": self.workload.autoscaler_target_utilization_percent,
                    "cooldown_seconds": self.workload.autoscaler_cooldown_seconds,
                    "current_capacity_units": self.workload.autoscaler_current_capacity_units,
                    "effective_server_limit": self.workload.autoscaler_effective_server_limit,
                    "last_scale_action_time": self.workload.autoscaler_last_scale_action_time,
                    "status": self.workload.autoscaler_status,
                },
                "telemetry_freshness": {
                    "metrics_last_updated_sim_time_seconds": self.workload.metrics_last_updated_sim_time_seconds,
                    "logs_last_updated_sim_time_seconds": self.workload.logs_last_updated_sim_time_seconds,
                    "telemetry_lag_seconds": self.workload.telemetry_lag_seconds,
                    "metrics_missing_ratio": self.workload.metrics_missing_ratio,
                    "logs_missing_ratio": self.workload.logs_missing_ratio,
                    "status": self.workload.telemetry_pipeline_status,
                },
                "load_balancer": {
                    "enabled": self.workload.load_balancer_enabled,
                    "backend_server_ids": list(self.workload.load_balancer_backend_server_ids),
                    "backend_weights": dict(self.workload.load_balancer_backend_weights),
                    "unhealthy_backend_ids": list(self.workload.load_balancer_unhealthy_backend_ids),
                    "routing_policy": self.workload.load_balancer_routing_policy,
                    "request_skew_ratio": self.workload.load_balancer_request_skew_ratio,
                    "unhealthy_routing_fraction": self.workload.load_balancer_unhealthy_routing_fraction,
                    "error_rate_percent": self.workload.load_balancer_error_rate_percent,
                },
                "trace_replay": dict(self.workload.trace_replay),
                "trace_replay_enabled": self.workload.trace_replay_enabled,
                "trace_replay_progress": dict(self.workload.trace_replay_progress),
                "job_duration_seconds": self.workload.job_duration_seconds,
                "job_started_at_sim_time_seconds": self.workload.job_started_at_sim_time_seconds,
                "job_completed": self.workload.job_completed,
                "throttle_rate_per_second": self.workload.throttle_rate_per_second,
                "noise_enabled": self.workload.noise_enabled,
                "noise_stddev": self.workload.noise_stddev,
            },
            "tenants": {
                tenant_id: {
                    "running": tenant.running,
                    "request_rate_per_second": tenant.request_rate_per_second,
                    "throttle_rate_per_second": tenant.throttle_rate_per_second,
                    "workload_class": tenant.workload_class,
                    "active_workload_class": tenant.active_workload_class,
                    "workload_profile_type": tenant.workload_profile_type,
                    "current_profile_type": tenant.current_profile_type,
                    "workload_profile_parameters": dict(tenant.workload_profile_parameters),
                    "placement_strategy": tenant.placement_strategy,
                    "target_rack_id": tenant.target_rack_id,
                    "priority": tenant.priority,
                    "quota_requests_per_second": tenant.quota_requests_per_second,
                    "max_server_count": tenant.max_server_count,
                    "trace_replay": dict(tenant.trace_replay),
                    "trace_replay_enabled": tenant.trace_replay_enabled,
                    "trace_replay_progress": dict(tenant.trace_replay_progress),
                    "job_duration_seconds": tenant.job_duration_seconds,
                    "job_started_at_sim_time_seconds": tenant.job_started_at_sim_time_seconds,
                    "job_completed": tenant.job_completed,
                }
                for tenant_id, tenant in sorted(self.tenants.items())
            },
            "cooling_units": [unit.to_dict() for unit in self.cooling_units],
            "active_controls": self.list_controls(),
            "supported_faults": sorted(SUPPORTED_FAULTS),
            "supported_actions": sorted(SUPPORTED_ACTIONS),
        }

    def state(self) -> dict[str, Any]:
        with self.lock:
            return {
                "sim_time_seconds": self.sim_time_seconds,
                "ambient_temperature_c": self.config.ambient_temperature_c,
                "total_it_power_kw": self.total_it_power_kw,
                "total_cooling_power_kw": self.total_cooling_power_kw,
                "facility_power_kw": self.facility_power_kw,
                "pue": self.pue,
                "sla_status": self.sla_status,
                "active_faults": [fault.to_dict() for fault in self.active_faults.values()],
                "active_controls": [action.to_dict() for action in self.controls],
                "workload": self.workload.to_dict(),
                "rooms": [room.to_dict() for room in self.rooms],
                "cooling_units": [unit.to_dict() for unit in self.cooling_units],
            }

    def state_summary(self) -> dict[str, Any]:
        with self.lock:
            average_cpu = sum(server.cpu_utilization_percent for server in self.servers) / max(len(self.servers), 1)
            average_rack_inlet = sum(rack.inlet_temperature_c for rack in self.racks) / max(len(self.racks), 1)
            average_reported_rack_inlet = sum(rack.reported_inlet_temperature_c for rack in self.racks) / max(len(self.racks), 1)
            max_rack_inlet = max((rack.inlet_temperature_c for rack in self.racks), default=0.0)
            max_reported_rack_inlet = max((rack.reported_inlet_temperature_c for rack in self.racks), default=0.0)
            max_rack_outlet = max((rack.outlet_temperature_c for rack in self.racks), default=0.0)
            sensor_disagreements = [
                *(rack.temperature_sensor_disagreement_c for rack in self.racks),
                *(server.temperature_sensor_disagreement_c for server in self.servers),
            ]
            throttle_factors = [server.thermal_throttle_factor for server in self.servers]
            packet_loss_racks = [rack for rack in self.racks if rack.network_packet_loss_percent > 0.0]
            max_network_packet_loss_percent = max(
                [self.workload.network_packet_loss_percent]
                + [rack.network_packet_loss_percent for rack in self.racks],
                default=0.0,
            )
            return {
                "sim_time_seconds": self.sim_time_seconds,
                "sla_status": self.sla_status,
                "total_it_power_kw": self.total_it_power_kw,
                "total_cooling_power_kw": self.total_cooling_power_kw,
                "facility_power_kw": self.facility_power_kw,
                "pue": self.pue,
                "average_cpu_utilization_percent": round(average_cpu, 4),
                "average_rack_inlet_temperature_c": round(average_rack_inlet, 4),
                "average_reported_rack_inlet_temperature_c": round(average_reported_rack_inlet, 4),
                "max_rack_inlet_temperature_c": round(max_rack_inlet, 4),
                "max_reported_rack_inlet_temperature_c": round(max_reported_rack_inlet, 4),
                "max_rack_outlet_temperature_c": round(max_rack_outlet, 4),
                "thermal_warnings": len([rack for rack in self.racks if rack.thermal_status == "warning"]),
                "thermal_critical": len([rack for rack in self.racks if rack.thermal_status == "critical"]),
                "power_overloaded_racks": len([rack for rack in self.racks if rack.power_status == "overloaded"]),
                "power_budget_violating_racks": len(
                    [rack for rack in self.racks if rack.power_budget_status == "violated"]
                ),
                "max_power_budget_utilization_ratio": round(
                    max((rack.total_power_kw / max(rack.power_budget_kw, 0.001) for rack in self.racks), default=0.0),
                    4,
                ),
                "failed_servers": len([server for server in self.servers if server.status == "failed"]),
                "host_health_flapping_count": len(
                    [server for server in self.servers if server.health_check_status == "flapping"]
                ),
                "host_health_status_change_count": sum(server.health_status_change_count for server in self.servers),
                "temperature_sensor_unhealthy_count": len(
                    [
                        component
                        for component in [*self.racks, *self.servers]
                        if component.temperature_sensor_status in {"biased", "untrusted"}
                    ]
                ),
                "temperature_sensor_untrusted_count": len(
                    [
                        component
                        for component in [*self.racks, *self.servers]
                        if component.temperature_sensor_status == "untrusted"
                    ]
                ),
                "max_temperature_sensor_disagreement_c": round(max(sensor_disagreements, default=0.0), 4),
                "thermal_throttled_servers": len(
                    [server for server in self.servers if server.thermal_throttle_factor < 0.999]
                ),
                "min_thermal_throttle_factor": round(min(throttle_factors, default=1.0), 4),
                "network_packet_loss_racks": len(packet_loss_racks),
                "max_network_packet_loss_percent": round(max_network_packet_loss_percent, 4),
                "network_retransmit_rate": self.workload.network_retransmit_rate,
                "network_error_rate": self.workload.network_error_rate,
                "affected_rack_id": self.workload.affected_rack_id,
                "autoscaler_enabled": self.workload.autoscaler_enabled,
                "autoscaler_min_capacity": self.workload.autoscaler_min_capacity,
                "autoscaler_max_capacity": self.workload.autoscaler_max_capacity,
                "autoscaler_target_utilization_percent": self.workload.autoscaler_target_utilization_percent,
                "autoscaler_cooldown_seconds": self.workload.autoscaler_cooldown_seconds,
                "autoscaler_current_capacity_units": self.workload.autoscaler_current_capacity_units,
                "autoscaler_effective_server_limit": self.workload.autoscaler_effective_server_limit,
                "autoscaler_last_scale_action_time": self.workload.autoscaler_last_scale_action_time,
                "autoscaler_status": self.workload.autoscaler_status,
                "metrics_last_updated_sim_time_seconds": self.workload.metrics_last_updated_sim_time_seconds,
                "logs_last_updated_sim_time_seconds": self.workload.logs_last_updated_sim_time_seconds,
                "telemetry_lag_seconds": self.workload.telemetry_lag_seconds,
                "metrics_missing_ratio": self.workload.metrics_missing_ratio,
                "logs_missing_ratio": self.workload.logs_missing_ratio,
                "telemetry_pipeline_status": self.workload.telemetry_pipeline_status,
                "placement_policy_status": self.workload.placement_policy_status,
                "placement_policy_target_rack_id": self.workload.placement_policy_target_rack_id,
                "placement_policy_violating_racks": self.workload.placement_policy_violating_racks,
                "workload_placement_imbalance_ratio": self.workload.workload_placement_imbalance_ratio,
                "load_balancer_enabled": self.workload.load_balancer_enabled,
                "load_balancer_backend_server_ids": list(self.workload.load_balancer_backend_server_ids),
                "load_balancer_backend_weights": dict(self.workload.load_balancer_backend_weights),
                "load_balancer_unhealthy_backend_ids": list(self.workload.load_balancer_unhealthy_backend_ids),
                "load_balancer_routing_policy": self.workload.load_balancer_routing_policy,
                "load_balancer_backend_skew_ratio": self.workload.load_balancer_request_skew_ratio,
                "load_balancer_unhealthy_routing_fraction": self.workload.load_balancer_unhealthy_routing_fraction,
                "load_balancer_error_rate_percent": self.workload.load_balancer_error_rate_percent,
                "active_faults": [
                    self._fault_telemetry(fault)
                    for fault in self.active_faults.values()
                ],
                "workload_type": self.workload.current_profile_type,
                "workload_class": self.workload.active_workload_class,
                "workload_running": self.workload.running,
                "workload_trace_replay_enabled": self.workload.trace_replay_enabled,
                "workload_trace_replay_progress": dict(self.workload.trace_replay_progress),
                "workload_active_tenant_count": self.workload.active_tenant_count,
                "tenant_summaries": {
                    tenant_id: tenant.to_dict() for tenant_id, tenant in sorted(self.tenants.items())
                },
                "workload_class_resource_demand": dict(self.workload.class_resource_demand),
                "workload_job_duration_seconds": self.workload.job_duration_seconds,
                "workload_job_started_at_sim_time_seconds": self.workload.job_started_at_sim_time_seconds,
                "workload_job_completed": self.workload.job_completed,
                "workload_allocated_server_ids": list(self.workload.allocated_server_ids),
                "workload_allocation_weights": dict(self.workload.allocation_weights),
                "workload_desired_allocated_server_ids": list(self.workload.desired_allocated_server_ids),
                "workload_desired_allocation_weights": dict(self.workload.desired_allocation_weights),
                "workload_cpu_demand": self.workload.cpu_demand,
                "workload_memory_demand": self.workload.memory_demand,
                "workload_network_demand": self.workload.network_demand,
                "workload_storage_demand": self.workload.storage_demand,
                "workload_gpu_demand": self.workload.gpu_demand,
                "workload_current_demand_per_second": (
                    self.workload.current_demand_per_second if self.workload.running else 0.0
                ),
                "workload_uncapped_demand_per_second": (
                    self.workload.uncapped_demand_per_second if self.workload.running else 0.0
                ),
                "workload_request_rate_per_second": (
                    self.workload.current_demand_per_second if self.workload.running else 0.0
                ),
                "workload_configured_request_rate_per_second": self.workload.request_rate_per_second,
                "workload_maintenance_window_active": self.workload.maintenance_window_active,
                "workload_max_server_count": self.workload.max_server_count,
                "workload_forbidden_rack_ids": list(self.workload.forbidden_rack_ids),
                "workload_queue_length": self.workload.queue_length,
                "workload_average_latency_ms": self.workload.average_latency_ms,
                "workload_p95_latency_ms": self.workload.p95_latency_ms,
                "workload_service_capacity_requests_per_second": self.workload.service_capacity_requests_per_second,
                "workload_network_demand_mbps": self.workload.network_demand_mbps,
                "workload_network_congestion_ratio": self.workload.network_congestion_ratio,
                "workload_network_latency_penalty_ms": self.workload.network_latency_penalty_ms,
                "workload_network_packet_loss_percent": self.workload.network_packet_loss_percent,
                "workload_network_retransmit_rate": self.workload.network_retransmit_rate,
                "workload_network_error_rate": self.workload.network_error_rate,
                "workload_network_affected_rack_id": self.workload.affected_rack_id,
                "workload_storage_demand_iops": self.workload.storage_demand_iops,
                "workload_storage_utilization_ratio": self.workload.storage_utilization_ratio,
                "workload_storage_latency_penalty_ms": self.workload.storage_latency_penalty_ms,
                "workload_application_error_rate_percent": self.workload.application_error_rate_percent,
                "workload_dropped_requests_per_second": self.workload.dropped_requests_per_second,
                "workload_gpu_utilization_percent": self.workload.gpu_utilization_percent,
                "workload_queueing_latency_ms": self.workload.queueing_latency_ms,
                "workload_service_time_latency_ms": self.workload.service_time_latency_ms,
                "control_plane_scheduler_api_latency_ms": self.workload.scheduler_api_latency_ms,
                "control_plane_scheduler_pending_operations": self.workload.scheduler_pending_operations,
            }

    def query_metric_history(
        self,
        start_time_seconds: int | float,
        end_time_seconds: int | float,
        entity_id: str | None = None,
        entity_type: str | None = None,
        metric_name: str | None = None,
    ) -> list[MetricSeries]:
        start_time = _coerce_query_time("start_time_seconds", start_time_seconds)
        end_time = _coerce_query_time("end_time_seconds", end_time_seconds)
        if start_time > end_time:
            raise SimulationError("start_time_seconds must be <= end_time_seconds")
        with self.lock:
            return metric_series_from_points(
                list(self._metric_history_points),
                start_time,
                end_time,
                entity_id=entity_id,
                entity_type=entity_type,
                metric_name=metric_name,
            )

    def list_faults(self) -> list[dict[str, Any]]:
        return [fault.to_dict() for fault in self.active_faults.values()]

    def get_fault(self, fault_id: str) -> dict[str, Any]:
        if fault_id not in self.active_faults:
            raise SimulationError(f"fault not found: {fault_id}")
        return self.active_faults[fault_id].to_dict()

    def list_controls(self) -> list[dict[str, Any]]:
        return [action.to_dict() for action in self.controls]


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _coerce_query_time(name: str, value: Any) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SimulationError(f"{name} must be numeric")
    if value < 0:
        raise SimulationError(f"{name} must be >= 0")
    return value


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _weighted_average(tenants: list[TenantWorkloadState], field_name: str) -> float:
    total_demand = sum(tenant.current_demand_per_second for tenant in tenants)
    if total_demand <= 0.0:
        return 0.0
    return sum(getattr(tenant, field_name) * tenant.current_demand_per_second for tenant in tenants) / total_demand


def _alert(
    alert_type: str,
    severity: str,
    target: str,
    message: str,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "alert_type": alert_type,
        "severity": severity,
        "target": target,
        "message": message,
        "details": details or {},
    }


def _structured_metrics(
    summary: dict[str, Any],
    racks: list[Rack],
    cooling_units: list[CoolingUnit],
    controls: list[ControlAction],
) -> dict[str, Any]:
    return {
        "facility": {
            "total_it_power_kw": summary["total_it_power_kw"],
            "total_cooling_power_kw": summary["total_cooling_power_kw"],
            "facility_power_kw": summary["facility_power_kw"],
            "pue": summary["pue"],
        },
        "thermal": {
            "average_rack_inlet_temperature_c": summary["average_rack_inlet_temperature_c"],
            "average_reported_rack_inlet_temperature_c": summary["average_reported_rack_inlet_temperature_c"],
            "max_rack_inlet_temperature_c": summary["max_rack_inlet_temperature_c"],
            "max_reported_rack_inlet_temperature_c": summary["max_reported_rack_inlet_temperature_c"],
            "max_rack_outlet_temperature_c": summary["max_rack_outlet_temperature_c"],
            "thermal_warnings": summary["thermal_warnings"],
            "thermal_critical": summary["thermal_critical"],
            "temperature_sensor_unhealthy_count": summary["temperature_sensor_unhealthy_count"],
            "max_temperature_sensor_disagreement_c": summary["max_temperature_sensor_disagreement_c"],
            "thermal_throttled_servers": summary["thermal_throttled_servers"],
            "min_thermal_throttle_factor": summary["min_thermal_throttle_factor"],
        },
        "power": {
            "power_overloaded_racks": summary["power_overloaded_racks"],
            "power_budget_violating_racks": summary["power_budget_violating_racks"],
            "max_power_budget_utilization_ratio": summary["max_power_budget_utilization_ratio"],
            "failed_servers": summary["failed_servers"],
            "average_cpu_utilization_percent": summary["average_cpu_utilization_percent"],
        },
        "workload": {
            "running": summary["workload_running"],
            "type": summary["workload_type"],
            "class": summary["workload_class"],
            "active_tenant_count": summary["workload_active_tenant_count"],
            "current_demand_per_second": summary["workload_current_demand_per_second"],
            "uncapped_demand_per_second": summary["workload_uncapped_demand_per_second"],
            "configured_request_rate_per_second": summary["workload_configured_request_rate_per_second"],
            "queue_length": summary["workload_queue_length"],
            "average_latency_ms": summary["workload_average_latency_ms"],
            "p95_latency_ms": summary["workload_p95_latency_ms"],
            "service_capacity_requests_per_second": summary["workload_service_capacity_requests_per_second"],
            "network_demand_mbps": summary["workload_network_demand_mbps"],
            "network_congestion_ratio": summary["workload_network_congestion_ratio"],
            "network_packet_loss_percent": summary["workload_network_packet_loss_percent"],
            "network_retransmit_rate": summary["workload_network_retransmit_rate"],
            "network_error_rate": summary["workload_network_error_rate"],
            "affected_rack_id": summary["workload_network_affected_rack_id"],
            "storage_demand_iops": summary["workload_storage_demand_iops"],
            "storage_utilization_ratio": summary["workload_storage_utilization_ratio"],
            "application_error_rate_percent": summary["workload_application_error_rate_percent"],
            "dropped_requests_per_second": summary["workload_dropped_requests_per_second"],
            "load_balancer_backend_skew_ratio": summary["load_balancer_backend_skew_ratio"],
            "load_balancer_unhealthy_routing_fraction": summary["load_balancer_unhealthy_routing_fraction"],
            "load_balancer_error_rate_percent": summary["load_balancer_error_rate_percent"],
            "gpu_utilization_percent": summary["workload_gpu_utilization_percent"],
            "allocated_server_count": len(summary["workload_allocated_server_ids"]),
        },
        "control_plane": {
            "scheduler_api_latency_ms": summary[
                "control_plane_scheduler_api_latency_ms"
            ],
            "scheduler_pending_operations": summary[
                "control_plane_scheduler_pending_operations"
            ],
            "autoscaler_enabled": summary["autoscaler_enabled"],
            "autoscaler_min_capacity": summary["autoscaler_min_capacity"],
            "autoscaler_max_capacity": summary["autoscaler_max_capacity"],
            "autoscaler_target_utilization_percent": summary["autoscaler_target_utilization_percent"],
            "autoscaler_cooldown_seconds": summary["autoscaler_cooldown_seconds"],
            "autoscaler_current_capacity_units": summary["autoscaler_current_capacity_units"],
            "autoscaler_effective_server_limit": summary["autoscaler_effective_server_limit"],
            "autoscaler_status": summary["autoscaler_status"],
            "placement_policy_status": summary["placement_policy_status"],
            "placement_policy_target_rack_id": summary["placement_policy_target_rack_id"],
            "placement_policy_violating_racks": summary["placement_policy_violating_racks"],
            "workload_placement_imbalance_ratio": summary["workload_placement_imbalance_ratio"],
        },
        "observability": {
            "metrics_last_updated_sim_time_seconds": summary["metrics_last_updated_sim_time_seconds"],
            "logs_last_updated_sim_time_seconds": summary["logs_last_updated_sim_time_seconds"],
            "telemetry_lag_seconds": summary["telemetry_lag_seconds"],
            "metrics_missing_ratio": summary["metrics_missing_ratio"],
            "logs_missing_ratio": summary["logs_missing_ratio"],
            "telemetry_pipeline_status": summary["telemetry_pipeline_status"],
        },
        "faults": {
            "active_count": len(summary["active_faults"]),
            "active": summary["active_faults"],
        },
        "racks": {
            rack.rack_id: {
                "row_id": rack.row_id,
                "inlet_temperature_c": rack.inlet_temperature_c,
                "outlet_temperature_c": rack.outlet_temperature_c,
                "power_kw": rack.total_power_kw,
                "power_budget_kw": rack.power_budget_kw,
                "thermal_status": rack.thermal_status,
                "power_status": rack.power_status,
                "power_budget_status": rack.power_budget_status,
                "network_packet_loss_percent": rack.network_packet_loss_percent,
                "network_retransmit_rate": rack.network_retransmit_rate,
                "network_error_rate": rack.network_error_rate,
                "network_path_status": rack.network_path_status,
                "reported_inlet_temperature_c": rack.reported_inlet_temperature_c,
                "temperature_sensor_status": rack.temperature_sensor_status,
                "temperature_sensor_disagreement_c": rack.temperature_sensor_disagreement_c,
                "thermal_throttle_factor": rack.thermal_throttle_factor,
                "average_cpu_utilization_percent": rack.average_cpu_utilization_percent,
            }
            for rack in racks
        },
        "servers": {
            server.server_id: {
                "rack_id": rack.rack_id,
                "status": server.status,
                "failed": server.status == "failed",
                "cpu_utilization_percent": server.cpu_utilization_percent,
                "memory_utilization_percent": server.memory_utilization_percent,
                "gpu_utilization_percent": server.gpu_utilization_percent,
                "power_kw": server.power_kw,
                "temperature_c": server.temperature_c,
                "reported_temperature_c": server.reported_temperature_c,
                "temperature_sensor_status": server.temperature_sensor_status,
                "temperature_sensor_disagreement_c": server.temperature_sensor_disagreement_c,
                "thermal_throttle_factor": server.thermal_throttle_factor,
                "cpu_frequency_scale": server.cpu_frequency_scale,
                "health_check_status": server.health_check_status,
                "health_status_change_count": server.health_status_change_count,
                "workload_assigned": server.workload_assigned,
            }
            for rack in racks
            for server in rack.servers
        },
        "cooling_units": {
            unit.cooling_unit_id: {
                "status": unit.status,
                "cooling_capacity_kw": unit.cooling_capacity_kw,
                "baseline_capacity_kw": unit.baseline_capacity_kw,
                "supply_air_temperature_c": unit.supply_air_temperature_c,
                "fan_speed_percent": unit.fan_speed_percent,
            }
            for unit in cooling_units
        },
        "controls": [control.to_dict() for control in controls],
        "tenants": summary["tenant_summaries"],
    }
