import json

import pytest

from aiopslab.orchestrator.problems.data_center_twin.visibility import assert_no_agent_leakage
from dc_twin.agent_interface import (
    AgentActionRequest,
    action_space,
    agent_observation,
    apply_agent_action,
    get_evaluator_state,
)
from dc_twin.config import default_config
from dc_twin.faults import FaultRequest, SUPPORTED_FAULTS
from dc_twin.simulator import DataCenterSimulator


FAULT_TARGETS = {
    "application_error": "application",
    "autoscaler_misconfiguration": "autoscaler",
    "control_plane_degradation": "control-plane",
    "cooling_degradation": "cooling-unit-1",
    "intermittent_server_failure": "server-r1-row1-rack02-03",
    "load_balancer_misconfiguration": "load-balancer",
    "network_congestion_burst": "workload",
    "network_partition": "rack-r1-row1-01",
    "monitoring_pipeline_failure": "monitoring-pipeline",
    "placement_policy_misconfiguration": "rack-r1-row1-04",
    "power_overload": "rack-r1-row1-01",
    "power_budget_violation": "rack-r1-row1-02",
    "rack_hotspot": "rack-r1-row1-01",
    "server_failure": "server-r1-row1-rack01-01",
    "storage_io_saturation": "storage",
    "thermal_sensor_miscalibration": "rack-r1-row1-02",
    "thermal_throttling": "rack-r1-row1-02",
    "tor_packet_loss": "rack-r1-row1-03",
}
FORBIDDEN_KEYS = {
    "active_faults",
    "active_faults_after",
    "active_faults_before",
    "fault_id",
    "fault_summary",
    "fault_type",
    "incident_domains",
    "initial_seed",
    "reset_config",
    "scenario_setup",
    "score_hints",
    "seed",
    "supported_faults",
    "desired_allocated_server_ids",
    "desired_allocation_weights",
    "workload_desired_allocated_server_ids",
    "workload_desired_allocation_weights",
}
FORBIDDEN_SUBSTRINGS = {
    "active_fault",
    "active server_failure",
    "application_error",
    "autoscaler_misconfiguration",
    "base_latency_ms",
    "control_plane_degradation",
    "cooling_degradation",
    "cpu_cost_per_request",
    "fault_injected",
    "fault_latency_penalty_ms",
    "intermittent_server_failure",
    "latency_fault_penalty_ms",
    "load_balancer_misconfiguration",
    "memory_cost_per_request_mb",
    "network_base_latency_ms",
    "network_congestion_burst",
    "network_capacity_mbps",
    "network_congestion_penalty_ms",
    "network_partition",
    "noise_enabled",
    "noise_stddev",
    "monitoring_pipeline_failure",
    "placement_policy_misconfiguration",
    "power_overload",
    "power_budget_violation",
    "rack_hotspot",
    "server_failure injected",
    "server_failure",
    "service_latency_ms",
    "storage_base_latency_ms",
    "storage_capacity_iops",
    "storage_congestion_penalty_ms",
    "storage_io_saturation",
    "thermal_sensor_miscalibration",
    "thermal_throttling",
    "tor_packet_loss",
    "workload_network_latency_penalty_ms",
    "workload_storage_latency_penalty_ms",
}


def _keys(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from _keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from _keys(item)


def _agent_response_for_fault(fault_type):
    sim = DataCenterSimulator(default_config())
    sim.reset(seed=42, config_override={"simulation": {"auto_advance": False}})
    sim.start_workload({"request_rate_per_second": 900, "noise_enabled": False})
    sim.step(1)
    sim.inject_fault(
        FaultRequest(
            fault_type=fault_type,
            target=FAULT_TARGETS[fault_type],
            severity=1.0,
            duration_seconds=300,
        )
    )
    sim.step(1)
    return sim, apply_agent_action(
        sim,
        AgentActionRequest(action_type="observe", log_limit=20, include_config=True),
    )


@pytest.mark.parametrize("fault_type", sorted(SUPPORTED_FAULTS))
def test_agent_visible_response_hides_fault_oracle_fields_for_all_faults(fault_type):
    assert set(FAULT_TARGETS) == SUPPORTED_FAULTS
    sim, response = _agent_response_for_fault(fault_type)
    observation = response["observation"]
    serialized = json.dumps(response, sort_keys=True, default=str).lower()

    assert FORBIDDEN_KEYS.isdisjoint(set(_keys(response)))
    assert all(substring not in serialized for substring in FORBIDDEN_SUBSTRINGS)
    assert "server-" not in serialized
    assert response["accepted"] is True
    assert "score_hints" not in response
    assert "active_faults_before" not in response
    assert "active_faults_after" not in response

    assert observation["episode_id"]
    assert "sim_time_seconds" in observation
    assert "sla_status" in observation
    assert "summary" in observation
    assert "alerts" in observation
    assert observation["action_schema_ref"] == "/agent/action-space"
    assert "available_actions" not in observation
    assert "workload_average_latency_ms" in observation["summary"]
    assert "workload_p95_latency_ms" in observation["summary"]
    assert "facility_power_kw" in observation["summary"]
    assert "max_rack_inlet_temperature_c" in observation["summary"]
    assert "workload_desired_allocated_server_ids" not in observation["summary"]
    assert "workload_desired_allocation_weights" not in observation["summary"]
    assert "desired_allocated_server_ids" not in serialized
    assert "desired_allocation_weights" not in serialized
    assert "available_actions" not in response
    assert response["action_schema_ref"] == "/agent/action-space"
    assert_no_agent_leakage(response)

    evaluator_state = get_evaluator_state(sim)
    assert evaluator_state["active_faults"][0]["fault_type"] == fault_type


def test_agent_observation_hides_desired_allocation_and_fault_metadata():
    sim, response = _agent_response_for_fault("server_failure")
    observation = agent_observation(sim, log_limit=20, include_config=True)
    serialized = json.dumps(observation, sort_keys=True, default=str).lower()

    assert FORBIDDEN_KEYS.isdisjoint(set(_keys(observation)))
    assert all(substring not in serialized for substring in FORBIDDEN_SUBSTRINGS)
    assert "workload_desired_allocated_server_ids" not in observation["summary"]
    assert "workload_desired_allocation_weights" not in observation["summary"]
    assert "desired_allocated_server_ids" not in serialized
    assert "desired_allocation_weights" not in serialized
    assert "baseline_capacity_kw" not in json.dumps(observation.get("configuration", {}), sort_keys=True, default=str)
    assert "available_actions" not in observation
    assert observation["action_schema_ref"] == "/agent/action-space"
    assert_no_agent_leakage(response)
    assert_no_agent_leakage(observation)


def test_evaluator_visibility_retains_debug_allocation_and_fault_state():
    sim, _response = _agent_response_for_fault("server_failure")
    evaluator_state = get_evaluator_state(sim, log_limit=20, include_config=True)

    assert evaluator_state["active_faults"][0]["fault_type"] == "server_failure"
    assert evaluator_state["summary"]["active_faults"][0]["fault_type"] == "server_failure"
    assert "workload_desired_allocated_server_ids" in evaluator_state["summary"]
    assert "workload_desired_allocation_weights" in evaluator_state["summary"]
    assert "domain_contract" in evaluator_state["available_actions"]


def test_agent_action_schema_is_canonical_and_opt_in_for_responses():
    sim, _response = _agent_response_for_fault("server_failure")

    schema = action_space()
    compact = apply_agent_action(sim, AgentActionRequest(action_type="observe", include_config=True))
    explicit = apply_agent_action(
        sim,
        AgentActionRequest(action_type="observe", include_config=True, include_action_schema=True),
    )

    compact_text = json.dumps(compact, sort_keys=True, default=str)
    explicit_text = json.dumps(explicit, sort_keys=True, default=str)
    assert "control_actions" in schema
    assert_no_agent_leakage(schema)
    assert "available_actions" not in compact
    assert "available_actions" not in compact["observation"]
    assert compact["action_schema_ref"] == "/agent/action-space"
    assert compact["observation"]["action_schema_ref"] == "/agent/action-space"
    assert "available_actions" in explicit
    assert "available_actions" in explicit["observation"]
    assert len(compact_text) < len(explicit_text) * 0.85


def test_agent_host_visibility_defaults_to_rack_and_keeps_debug_exact_target():
    sim, response = _agent_response_for_fault("server_failure")
    observation = response["observation"]
    alert_text = json.dumps(observation["alerts"], sort_keys=True, default=str)

    assert "server-r1-row1-rack01-01" not in alert_text
    assert "rack-r1-row1-01" in alert_text
    assert any(
        alert.get("alert_type") == "HostHealthCheckFailed"
        and alert.get("target") == "rack-r1-row1-01"
        and alert.get("details", {}).get("affected_host_count") == 1
        for alert in observation["alerts"]
    )
    summary_text = json.dumps(observation["summary"], sort_keys=True, default=str)
    assert "server-r1-row1-rack01-01" not in summary_text
    assert "workload_allocated_server_ids" not in observation["summary"]
    assert "workload_allocation_weights" not in observation["summary"]
    assert "workload_allocated_rack_ids" in observation["summary"]
    assert "workload_allocation_weights_by_rack" in observation["summary"]

    host_response = apply_agent_action(
        sim,
        AgentActionRequest(action_type="observe", include_config=True, host_visibility="host"),
    )
    assert "server-r1-row1-rack01-01" in json.dumps(host_response["observation"]["alerts"], sort_keys=True, default=str)

    evaluator_state = get_evaluator_state(sim)
    assert evaluator_state["active_faults"][0]["target"] == "server-r1-row1-rack01-01"


def test_rack_visibility_before_after_diff_cannot_reveal_failed_host():
    simulator = DataCenterSimulator(default_config())
    simulator.reset(seed=42, config_override={"simulation": {"auto_advance": False}})
    simulator.start_workload({"request_rate_per_second": 900, "noise_enabled": False})
    simulator.step(1)
    before = agent_observation(simulator)
    simulator.inject_fault(
        FaultRequest(
            fault_type="server_failure",
            target="server-r1-row1-rack01-01",
            severity=1.0,
            duration_seconds=300,
        )
    )
    after = agent_observation(simulator)

    for payload in (before, after):
        rendered = json.dumps(payload, sort_keys=True, default=str)
        assert "server-r1-row1-rack01-01" not in rendered
        assert "workload_allocated_server_ids" not in payload["summary"]
        assert "workload_allocation_weights" not in payload["summary"]
        assert payload["summary"]["workload_allocated_rack_ids"]


def test_rack_visibility_coarsens_maintenance_profile_host_ids_recursively():
    simulator = DataCenterSimulator(default_config())
    simulator.start_workload(
        {
            "request_rate_per_second": 500,
            "workload_profile_type": "maintenance_window",
            "workload_profile_parameters": {
                "baseline_rate_per_second": 500,
                "maintenance_start_time_seconds": 0,
                "maintenance_end_time_seconds": 10,
                "affected_server_ids": ["server-r1-row1-rack01-01"],
                "affected_server_workload_fraction": 0.75,
            },
            "noise_enabled": False,
        }
    )
    simulator.step()

    observation = agent_observation(simulator, log_limit=20, include_config=True)
    rendered = json.dumps(observation, sort_keys=True, default=str)
    profile = observation["configuration"]["workload"][
        "workload_profile_parameters"
    ]

    assert "server-r1-row1-rack01-01" not in rendered
    assert profile["affected_rack_ids"] == ["rack-r1-row1-01"]
    assert profile["affected_host_counts_by_rack"] == {"rack-r1-row1-01": 1}


def test_server_maintenance_action_descriptions_do_not_imply_repair_or_oracle():
    schema = action_space()
    clear_description = schema["control_actions"]["clear_server_maintenance"]["description"]
    set_description = schema["control_actions"]["set_server_maintenance"]["description"]

    assert "does not repair failed hardware" in clear_description
    assert "Return a maintenance server to healthy state" not in clear_description
    assert "not a diagnostic oracle" in set_description
    assert schema["control_actions"]["set_server_maintenance"][
        "required_any"
    ] == ["server_id", "rack_id"]
