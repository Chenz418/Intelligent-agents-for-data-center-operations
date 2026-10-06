"""Prometheus text exposition rendering."""

from __future__ import annotations

from dc_twin.simulator import DataCenterSimulator


def render_metrics(simulator: DataCenterSimulator) -> bytes:
    state = simulator.state_summary()
    lines: list[str] = []
    _gauge(lines, "dc_twin_sim_time_seconds", "Simulation time in seconds", state["sim_time_seconds"])
    _gauge(lines, "dc_twin_total_it_power_kw", "Total IT power draw", state["total_it_power_kw"])
    _gauge(lines, "dc_twin_total_cooling_power_kw", "Total cooling power draw", state["total_cooling_power_kw"])
    _gauge(lines, "dc_twin_facility_power_kw", "Facility power draw", state["facility_power_kw"])
    _gauge(lines, "dc_twin_pue", "Power usage effectiveness", state["pue"])
    _gauge(
        lines,
        "dc_twin_average_cpu_utilization_percent",
        "Average server CPU utilization",
        state["average_cpu_utilization_percent"],
    )
    _gauge(
        lines,
        "dc_twin_average_rack_inlet_temperature_c",
        "Average rack inlet temperature",
        state["average_rack_inlet_temperature_c"],
    )
    _gauge(
        lines,
        "dc_twin_max_rack_inlet_temperature_c",
        "Maximum rack inlet temperature",
        state["max_rack_inlet_temperature_c"],
    )
    _gauge(
        lines,
        "dc_twin_max_reported_rack_inlet_temperature_c",
        "Maximum reported rack inlet temperature",
        state["max_reported_rack_inlet_temperature_c"],
    )
    _gauge(
        lines,
        "dc_twin_max_temperature_sensor_disagreement_c",
        "Maximum reported-versus-physical temperature disagreement",
        state["max_temperature_sensor_disagreement_c"],
    )
    _gauge(
        lines,
        "dc_twin_power_budget_violating_racks",
        "Rack count above configured power budget",
        state["power_budget_violating_racks"],
    )
    _gauge(
        lines,
        "dc_twin_thermal_throttled_servers",
        "Server count with thermal capacity throttling",
        state["thermal_throttled_servers"],
    )
    _gauge(lines, "dc_twin_sla_violation", "SLA violation status", 1 if state["sla_status"] == "violated" else 0)
    _gauge(lines, "dc_twin_active_faults", "Active fault count", len(state["active_faults"]))
    alerts = simulator.alerts()
    _gauge(lines, "dc_twin_active_alerts", "Active observation alert count", len(alerts))
    for alert_labels, count in _alert_groups(alerts).items():
        _sample(
            lines,
            "dc_twin_alert_active",
            count,
            {"alert_type": alert_labels[0], "severity": alert_labels[1], "target": alert_labels[2]},
        )
    for fault_labels, values in _fault_groups(state["active_faults"]).items():
        labels = {
            "fault_type": fault_labels[0],
            "target": fault_labels[1],
            "status": fault_labels[2],
        }
        _sample(lines, "dc_twin_fault_active", values["count"], labels)
        _sample(lines, "dc_twin_fault_severity", values["max_severity"], labels)
        _sample(lines, "dc_twin_fault_min_severity", values["min_severity"], labels)
        _sample(lines, "dc_twin_fault_max_severity", values["max_severity"], labels)
        _sample(lines, "dc_twin_fault_duration_seconds", values["max_duration_seconds"], labels)
        _sample(lines, "dc_twin_fault_min_duration_seconds", values["min_duration_seconds"], labels)
        _sample(lines, "dc_twin_fault_max_duration_seconds", values["max_duration_seconds"], labels)
        _sample(lines, "dc_twin_fault_min_remaining_duration_seconds", values["min_remaining_duration_seconds"], labels)
        _sample(lines, "dc_twin_fault_max_remaining_duration_seconds", values["max_remaining_duration_seconds"], labels)
        _sample(lines, "dc_twin_fault_started_at_sim_time_seconds", values["started_at_sim_time_seconds"], labels)
        _sample(
            lines,
            "dc_twin_fault_last_started_at_sim_time_seconds",
            values["last_started_at_sim_time_seconds"],
            labels,
        )
    _gauge(
        lines,
        "dc_twin_workload_request_rate_per_second",
        "Current workload request rate",
        state["workload_request_rate_per_second"],
    )
    _gauge(
        lines,
        "dc_twin_workload_current_demand_per_second",
        "Current generated workload demand",
        state["workload_current_demand_per_second"],
    )
    workload_running = 1 if state["workload_running"] else 0
    _gauge(lines, "dc_twin_workload_running", "Workload running status", workload_running)
    _sample(
        lines,
        "dc_twin_workload_profile_active",
        workload_running,
        {"profile_type": state["workload_type"]},
    )
    _sample(
        lines,
        "dc_twin_workload_class_active",
        workload_running,
        {"workload_class": state["workload_class"]},
    )
    _gauge(lines, "dc_twin_workload_cpu_demand", "Normalized CPU demand for active workload class", state["workload_cpu_demand"])
    _gauge(
        lines,
        "dc_twin_workload_memory_demand",
        "Normalized memory demand for active workload class",
        state["workload_memory_demand"],
    )
    _gauge(
        lines,
        "dc_twin_workload_network_demand",
        "Normalized network demand for active workload class",
        state["workload_network_demand"],
    )
    _gauge(
        lines,
        "dc_twin_workload_storage_demand",
        "Normalized storage I/O demand for active workload class",
        state["workload_storage_demand"],
    )
    _gauge(lines, "dc_twin_workload_gpu_demand", "Normalized GPU demand for active workload class", state["workload_gpu_demand"])
    _gauge(
        lines,
        "dc_twin_workload_maintenance_window_active",
        "Workload maintenance window active status",
        1 if state["workload_maintenance_window_active"] else 0,
    )
    _gauge(lines, "dc_twin_workload_queue_length", "Unprocessed workload queue length", state["workload_queue_length"])
    _gauge(
        lines,
        "dc_twin_workload_average_latency_ms",
        "Estimated average request latency",
        state["workload_average_latency_ms"],
    )
    _gauge(
        lines,
        "dc_twin_workload_p95_latency_ms",
        "Estimated p95 request latency",
        state["workload_p95_latency_ms"],
    )
    _gauge(
        lines,
        "dc_twin_workload_network_congestion_ratio",
        "Workload network demand divided by available capacity",
        state["workload_network_congestion_ratio"],
    )
    _gauge(
        lines,
        "dc_twin_workload_network_demand_mbps",
        "Current workload network demand",
        state["workload_network_demand_mbps"],
    )
    _gauge(
        lines,
        "dc_twin_workload_network_latency_penalty_ms",
        "Estimated request latency added by network congestion",
        state["workload_network_latency_penalty_ms"],
    )
    _gauge(
        lines,
        "dc_twin_workload_network_packet_loss_percent",
        "Packet loss percentage impacting the active workload path",
        state["workload_network_packet_loss_percent"],
    )
    _gauge(
        lines,
        "dc_twin_workload_network_retransmit_rate",
        "Retransmit rate impacting the active workload path",
        state["workload_network_retransmit_rate"],
    )
    _gauge(
        lines,
        "dc_twin_workload_network_error_rate",
        "Network error rate impacting the active workload path",
        state["workload_network_error_rate"],
    )
    _gauge(
        lines,
        "dc_twin_workload_storage_demand_iops",
        "Current workload storage I/O demand",
        state["workload_storage_demand_iops"],
    )
    _gauge(
        lines,
        "dc_twin_workload_storage_utilization_ratio",
        "Workload storage I/O demand divided by available capacity",
        state["workload_storage_utilization_ratio"],
    )
    _gauge(
        lines,
        "dc_twin_workload_storage_latency_penalty_ms",
        "Estimated request latency added by storage I/O pressure",
        state["workload_storage_latency_penalty_ms"],
    )
    _gauge(
        lines,
        "dc_twin_workload_application_error_rate_percent",
        "Current application error rate induced by application faults",
        state["workload_application_error_rate_percent"],
    )
    _gauge(
        lines,
        "dc_twin_workload_dropped_requests_per_second",
        "Current failed application requests per second",
        state["workload_dropped_requests_per_second"],
    )
    _gauge(
        lines,
        "dc_twin_load_balancer_backend_skew_ratio",
        "Largest backend request share for the active load balancer",
        state["load_balancer_backend_skew_ratio"],
    )
    _gauge(
        lines,
        "dc_twin_load_balancer_unhealthy_routing_fraction",
        "Fraction of load-balanced requests routed to unhealthy backends",
        state["load_balancer_unhealthy_routing_fraction"],
    )
    _gauge(
        lines,
        "dc_twin_load_balancer_error_rate_percent",
        "Application error percentage induced by load balancer routing state",
        state["load_balancer_error_rate_percent"],
    )
    _gauge(
        lines,
        "dc_twin_workload_gpu_utilization_percent",
        "Average GPU utilization generated by workload class",
        state["workload_gpu_utilization_percent"],
    )
    _gauge(
        lines,
        "dc_twin_workload_active_tenants",
        "Active workload tenant count",
        state["workload_active_tenant_count"],
    )
    _gauge(
        lines,
        "dc_twin_autoscaler_effective_server_limit",
        "Effective server limit enforced by autoscaler policy",
        state["autoscaler_effective_server_limit"] or 0,
    )
    _gauge(
        lines,
        "dc_twin_autoscaler_current_capacity_units",
        "Current autoscaler capacity units",
        state["autoscaler_current_capacity_units"] or 0,
    )
    _gauge(
        lines,
        "dc_twin_autoscaler_target_utilization_percent",
        "Autoscaler target utilization",
        state["autoscaler_target_utilization_percent"],
    )
    _gauge(
        lines,
        "dc_twin_telemetry_lag_seconds",
        "Telemetry lag visible to operators",
        state["telemetry_lag_seconds"],
    )
    _gauge(
        lines,
        "dc_twin_metrics_missing_ratio",
        "Fraction of metrics missing from the monitoring pipeline",
        state["metrics_missing_ratio"],
    )
    _gauge(
        lines,
        "dc_twin_logs_missing_ratio",
        "Fraction of logs missing from the monitoring pipeline",
        state["logs_missing_ratio"],
    )
    _gauge(
        lines,
        "dc_twin_placement_policy_violating_racks",
        "Rack count violating placement policy",
        state["placement_policy_violating_racks"],
    )
    _gauge(
        lines,
        "dc_twin_workload_placement_imbalance_ratio",
        "Largest rack allocation share for the active workload",
        state["workload_placement_imbalance_ratio"],
    )
    _gauge(
        lines,
        "dc_twin_workload_allocated_servers",
        "Allocated server count for the single workload",
        len(state["workload_allocated_server_ids"]),
    )
    for server_id, weight in state["workload_allocation_weights"].items():
        _sample(
            lines,
            "dc_twin_workload_allocation_weight",
            weight,
            {"server_id": server_id},
        )
    for tenant_id, tenant in state["tenant_summaries"].items():
        labels = {"tenant_id": tenant_id, "workload_class": tenant["active_workload_class"]}
        _sample(
            lines,
            "dc_twin_workload_tenant_current_demand_per_second",
            tenant["current_demand_per_second"],
            labels,
        )
        _sample(
            lines,
            "dc_twin_workload_tenant_uncapped_demand_per_second",
            tenant.get("uncapped_demand_per_second", tenant["current_demand_per_second"]),
            labels,
        )
        if tenant.get("throttle_rate_per_second") is not None:
            _sample(
                lines,
                "dc_twin_workload_tenant_throttle_rate_per_second",
                tenant["throttle_rate_per_second"],
                labels,
            )
        _sample(lines, "dc_twin_workload_tenant_queue_length", tenant["queue_length"], labels)
        _sample(lines, "dc_twin_workload_tenant_average_latency_ms", tenant["average_latency_ms"], labels)
        _sample(lines, "dc_twin_workload_tenant_p95_latency_ms", tenant["p95_latency_ms"], labels)
        _sample(
            lines,
            "dc_twin_workload_tenant_allocated_servers",
            len(tenant["allocated_server_ids"]),
            labels,
        )
        for server_id, weight in tenant.get("allocation_weights", {}).items():
            _sample(
                lines,
                "dc_twin_workload_tenant_allocation_weight",
                weight,
                {**labels, "server_id": server_id},
            )
        _sample(lines, "dc_twin_workload_tenant_cpu_usage_percent", tenant["cpu_usage_percent"], labels)
        _sample(lines, "dc_twin_workload_tenant_network_demand_mbps", tenant["network_demand_mbps"], labels)
        _sample(
            lines,
            "dc_twin_workload_tenant_network_packet_loss_percent",
            tenant.get("network_packet_loss_percent", 0.0),
            labels,
        )
        _sample(
            lines,
            "dc_twin_workload_tenant_network_retransmit_rate",
            tenant.get("network_retransmit_rate", 0.0),
            labels,
        )
        _sample(
            lines,
            "dc_twin_workload_tenant_network_error_rate",
            tenant.get("network_error_rate", 0.0),
            labels,
        )
        _sample(lines, "dc_twin_workload_tenant_storage_demand_iops", tenant["storage_demand_iops"], labels)
        _sample(
            lines,
            "dc_twin_workload_tenant_application_error_rate_percent",
            tenant["application_error_rate_percent"],
            labels,
        )
        _sample(
            lines,
            "dc_twin_workload_tenant_dropped_requests_per_second",
            tenant["dropped_requests_per_second"],
            labels,
        )
        _sample(lines, "dc_twin_workload_tenant_gpu_utilization_percent", tenant["gpu_utilization_percent"], labels)
        _sample(lines, "dc_twin_workload_tenant_sla_violation_count", tenant["sla_violation_count"], labels)

    for rack in simulator.racks:
        _sample(lines, "dc_twin_rack_inlet_temperature_c", rack.inlet_temperature_c, {"rack_id": rack.rack_id})
        _sample(
            lines,
            "dc_twin_rack_reported_inlet_temperature_c",
            rack.reported_inlet_temperature_c,
            {"rack_id": rack.rack_id},
        )
        _sample(lines, "dc_twin_rack_outlet_temperature_c", rack.outlet_temperature_c, {"rack_id": rack.rack_id})
        _sample(lines, "dc_twin_rack_power_kw", rack.total_power_kw, {"rack_id": rack.rack_id})
        _sample(lines, "dc_twin_rack_power_budget_kw", rack.power_budget_kw, {"rack_id": rack.rack_id})
        _sample(
            lines,
            "dc_twin_rack_network_packet_loss_percent",
            rack.network_packet_loss_percent,
            {"rack_id": rack.rack_id},
        )
        _sample(
            lines,
            "dc_twin_rack_network_retransmit_rate",
            rack.network_retransmit_rate,
            {"rack_id": rack.rack_id},
        )
        _sample(
            lines,
            "dc_twin_rack_network_error_rate",
            rack.network_error_rate,
            {"rack_id": rack.rack_id},
        )
        _sample(
            lines,
            "dc_twin_rack_power_budget_violation",
            1 if rack.power_budget_status == "violated" else 0,
            {"rack_id": rack.rack_id},
        )
        _sample(
            lines,
            "dc_twin_rack_temperature_sensor_disagreement_c",
            rack.temperature_sensor_disagreement_c,
            {"rack_id": rack.rack_id},
        )
        _sample(
            lines,
            "dc_twin_rack_thermal_throttle_factor",
            rack.thermal_throttle_factor,
            {"rack_id": rack.rack_id},
        )
        for status in ("normal", "warning", "critical"):
            _sample(
                lines,
                "dc_twin_rack_thermal_status",
                1 if rack.thermal_status == status else 0,
                {"rack_id": rack.rack_id, "status": status},
            )
        for server in rack.servers:
            labels = {"server_id": server.server_id, "rack_id": rack.rack_id}
            _sample(lines, "dc_twin_server_cpu_utilization_percent", server.cpu_utilization_percent, labels)
            _sample(lines, "dc_twin_server_power_kw", server.power_kw, labels)
            _sample(lines, "dc_twin_server_temperature_c", server.temperature_c, labels)
            _sample(lines, "dc_twin_server_reported_temperature_c", server.reported_temperature_c, labels)
            _sample(
                lines,
                "dc_twin_server_temperature_sensor_disagreement_c",
                server.temperature_sensor_disagreement_c,
                labels,
            )
            _sample(lines, "dc_twin_server_thermal_throttle_factor", server.thermal_throttle_factor, labels)
            _sample(lines, "dc_twin_server_health_status_changes", server.health_status_change_count, labels)
            _sample(lines, "dc_twin_server_failed", 1 if server.status == "failed" else 0, labels)
    for unit in simulator.cooling_units:
        _sample(
            lines,
            "dc_twin_cooling_capacity_kw",
            unit.cooling_capacity_kw,
            {"cooling_unit_id": unit.cooling_unit_id},
        )
    return ("\n".join(lines) + "\n").encode("utf-8")


def _gauge(lines: list[str], name: str, help_text: str, value: float) -> None:
    lines.append(f"# HELP {name} {help_text}")
    lines.append(f"# TYPE {name} gauge")
    _sample(lines, name, value, {})


def _fault_groups(active_faults: list[dict]) -> dict[tuple[str, str, str], dict[str, float]]:
    groups: dict[tuple[str, str, str], dict[str, float]] = {}
    for fault in active_faults:
        key = (fault["fault_type"], fault["target"], fault["status"])
        severity = float(fault["severity"])
        duration_seconds = float(fault["duration_seconds"])
        remaining_duration_seconds = float(fault.get("remaining_duration_seconds", duration_seconds))
        started_at = float(fault["started_at_sim_time_seconds"])
        values = groups.setdefault(
            key,
            {
                "count": 0.0,
                "min_severity": severity,
                "max_severity": severity,
                "min_duration_seconds": duration_seconds,
                "max_duration_seconds": duration_seconds,
                "min_remaining_duration_seconds": remaining_duration_seconds,
                "max_remaining_duration_seconds": remaining_duration_seconds,
                "started_at_sim_time_seconds": started_at,
                "last_started_at_sim_time_seconds": started_at,
            },
        )
        values["count"] += 1.0
        values["min_severity"] = min(values["min_severity"], severity)
        values["max_severity"] = max(values["max_severity"], severity)
        values["min_duration_seconds"] = min(values["min_duration_seconds"], duration_seconds)
        values["max_duration_seconds"] = max(values["max_duration_seconds"], duration_seconds)
        values["min_remaining_duration_seconds"] = min(
            values["min_remaining_duration_seconds"],
            remaining_duration_seconds,
        )
        values["max_remaining_duration_seconds"] = max(
            values["max_remaining_duration_seconds"],
            remaining_duration_seconds,
        )
        values["started_at_sim_time_seconds"] = min(
            values["started_at_sim_time_seconds"],
            started_at,
        )
        values["last_started_at_sim_time_seconds"] = max(
            values["last_started_at_sim_time_seconds"],
            started_at,
        )
    return groups


def _alert_groups(alerts: list[dict]) -> dict[tuple[str, str, str], float]:
    groups: dict[tuple[str, str, str], float] = {}
    for alert in alerts:
        key = (alert["alert_type"], alert["severity"], alert["target"])
        groups[key] = groups.get(key, 0.0) + 1.0
    return groups


def _sample(lines: list[str], name: str, value: float, labels: dict[str, str]) -> None:
    if labels:
        label_text = ",".join(f'{key}="{_escape_label(str(val))}"' for key, val in labels.items())
        lines.append(f"{name}{{{label_text}}} {float(value)}")
    else:
        lines.append(f"{name} {float(value)}")


def _escape_label(value: str) -> str:
    return value.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')
