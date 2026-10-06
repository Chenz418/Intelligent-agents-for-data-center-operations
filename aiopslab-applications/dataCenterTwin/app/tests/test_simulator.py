import importlib.util
import json
import logging
from pathlib import Path
import sys

import pytest

from aiopslab.orchestrator.problems.data_center_twin.visibility import (
    assert_no_agent_leakage,
)
from dc_twin.agent_interface import (
    AgentActionRequest,
    action_space,
    agent_observation,
    apply_agent_action,
)
from dc_twin.config import default_config
from dc_twin.controls import ControlRequest, SUPPORTED_ACTIONS
from dc_twin.faults import FaultRequest
from dc_twin.metrics import render_metrics
from dc_twin.simulator import DataCenterSimulator


FIXTURES = Path(__file__).parent / "fixtures"
REPO_ROOT = Path(__file__).resolve().parents[4]


def load_data_center_twin_scenarios():
    scenario_path = (
        REPO_ROOT
        / "aiopslab"
        / "orchestrator"
        / "problems"
        / "data_center_twin"
        / "scenarios.py"
    )
    spec = importlib.util.spec_from_file_location(
        "test_data_center_twin_scenarios", scenario_path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


SCENARIO_MODULE = load_data_center_twin_scenarios()
BENCHMARK_SCENARIOS = list(SCENARIO_MODULE.SCENARIOS.values())
REPRESENTATIVE_SCENARIOS_BY_FAULT = [
    sorted(
        [
            scenario
            for scenario in BENCHMARK_SCENARIOS
            if scenario.fault_type == fault_type
        ],
        key=lambda scenario: scenario.problem_id,
    )[0]
    for fault_type in sorted({scenario.fault_type for scenario in BENCHMARK_SCENARIOS})
]
AGENT_VISIBLE_SYMPTOM_IGNORE_FRAGMENTS = (
    "episode_id",
    "last_updated",
    "sequence_id",
    "sim_time_seconds",
)
OBSERVABLE_FIELD_ALIASES = {
    "application_error_rate_percent": (
        "workload_application_error_rate_percent",
        "workload_error_rate_percent",
    ),
    "cooling_units": ("total_cooling_power_kw",),
    "facility": (
        "facility_power_kw",
        "pue",
        "total_cooling_power_kw",
        "total_it_power_kw",
    ),
    "network_packet_loss_percent": (
        "max_network_packet_loss_percent",
        "workload_network_packet_loss_percent",
    ),
    "power": ("facility_power_kw", "power_budget", "power_limit", "total_it_power_kw"),
    "racks": ("rack", "thermal", "power_budget", "power_limit"),
    "servers": ("failed_servers", "host_health", "workload_allocated_server_ids"),
    "storage": ("workload_storage",),
    "tenant_summaries": ("tenant_summaries",),
    "thermal": ("rack_inlet_temperature", "rack_outlet_temperature", "thermal"),
    "workload": ("workload",),
}


def make_sim(auto_advance=False):
    config = default_config().merged({"simulation": {"auto_advance": auto_advance}})
    return DataCenterSimulator(config)


def tenant_sim(tenants):
    config = default_config().merged(
        {
            "simulation": {"auto_advance": False},
            "topology": {
                "rooms": 1,
                "rows_per_room": 1,
                "racks_per_row": 1,
                "servers_per_rack": 4,
            },
            "thresholds": {"workload_queue_sla_threshold": 10},
        }
    )
    sim = DataCenterSimulator(config)
    sim.reset(seed=7)
    sim.start_workload(
        {
            "tenants": tenants,
            "network_capacity_mbps": 50,
            "storage_capacity_iops": 1000,
            "noise_enabled": False,
        }
    )
    return sim


def run_benchmark_scenario(scenario):
    sim = DataCenterSimulator(default_config())
    sim.reset(seed=scenario.seed, config_override=scenario.config_override)
    sim.start_workload(scenario.workload)
    if scenario.stabilization_ticks:
        sim.step(ticks=scenario.stabilization_ticks)
    before_fault = sim.state_summary()
    fault = sim.inject_fault(
        FaultRequest(
            fault_type=scenario.fault_type,
            target=scenario.fault_target,
            severity=scenario.fault_severity,
            duration_seconds=scenario.fault_duration_seconds,
        )
    )
    sim._benchmark_fault_started_at = fault["started_at_sim_time_seconds"]
    post_fault = sim.step(ticks=1)
    observation = sim.observation(log_limit=20, include_config=True)
    return sim, before_fault, post_fault, observation


def make_benchmark_simulator(scenario):
    sim = DataCenterSimulator(default_config())
    sim.reset(seed=scenario.seed, config_override=scenario.config_override)
    sim.start_workload(scenario.workload)
    if scenario.stabilization_ticks:
        sim.step(ticks=scenario.stabilization_ticks)
    return sim


def inject_scenario_fault(sim, scenario):
    return sim.inject_fault(
        FaultRequest(
            fault_type=scenario.fault_type,
            target=scenario.fault_target,
            severity=scenario.fault_severity,
            duration_seconds=scenario.fault_duration_seconds,
            **scenario.fault_parameters,
        )
    )


def agent_visible_symptom_snapshot(observation):
    return flatten_payload(
        {
            "summary": observation.get("summary", {}),
            "alerts": observation.get("alerts", []),
        }
    )


def flatten_payload(value, prefix=""):
    if isinstance(value, dict):
        flattened = {}
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else key
            flattened.update(flatten_payload(item, path))
        return flattened
    if isinstance(value, list):
        return {prefix: json.dumps(value, sort_keys=True, default=str)}
    return {prefix: value}


def changed_agent_visible_symptom_fields(before_observation, after_observation):
    before = agent_visible_symptom_snapshot(before_observation)
    after = agent_visible_symptom_snapshot(after_observation)
    changed = []
    for key in sorted(set(before) | set(after)):
        if any(fragment in key for fragment in AGENT_VISIBLE_SYMPTOM_IGNORE_FRAGMENTS):
            continue
        if before.get(key) != after.get(key):
            changed.append(key)
    return changed


def safe_signal_fields_for_fault(fault_type, changed_fields):
    incident_domains = action_space(visibility="evaluator")["domain_contract"][
        "incident_domains"
    ]
    observable_fields = incident_domains[fault_type]["observable_fields"]
    aliases = []
    for field in observable_fields:
        aliases.append(field)
        aliases.extend(OBSERVABLE_FIELD_ALIASES.get(field, ()))
    return [
        field
        for field in changed_fields
        if field.startswith("alerts") or any(alias in field for alias in aliases)
    ]


def summary_satisfies_success_criteria(summary, scenario, sim=None):
    criteria = scenario.success_criteria
    required_control_actions = set(criteria.get("required_control_actions") or [])
    if required_control_actions:
        if sim is None:
            return False
        applied_actions = {
            call["request"]["action_type"]
            for call in getattr(sim, "_benchmark_agent_actions", [])
            if agent_action_matches_declared_mitigation(call, scenario)
        }
        if not required_control_actions <= applied_actions:
            return False
    if (
        criteria.get("sla_status") is not None
        and summary.get("sla_status") != criteria["sla_status"]
    ):
        return False
    threshold_fields = {
        "thermal_critical_max": "thermal_critical",
        "power_overloaded_racks_max": "power_overloaded_racks",
        "failed_servers_max": "failed_servers",
        "workload_queue_length_max": "workload_queue_length",
        "workload_network_congestion_ratio_max": "workload_network_congestion_ratio",
        "workload_average_latency_ms_max": "workload_average_latency_ms",
        "workload_p95_latency_ms_max": "workload_p95_latency_ms",
        "workload_network_packet_loss_percent_max": "workload_network_packet_loss_percent",
        "workload_network_retransmit_rate_max": "workload_network_retransmit_rate",
        "workload_network_error_rate_max": "workload_network_error_rate",
        "max_network_packet_loss_percent_max": "max_network_packet_loss_percent",
        "network_retransmit_rate_max": "network_retransmit_rate",
        "network_error_rate_max": "network_error_rate",
        "telemetry_lag_seconds_max": "telemetry_lag_seconds",
        "metrics_missing_ratio_max": "metrics_missing_ratio",
        "logs_missing_ratio_max": "logs_missing_ratio",
        "placement_policy_violating_racks_max": "placement_policy_violating_racks",
        "workload_placement_imbalance_ratio_max": "workload_placement_imbalance_ratio",
        "load_balancer_backend_skew_ratio_max": "load_balancer_backend_skew_ratio",
        "load_balancer_unhealthy_routing_fraction_max": "load_balancer_unhealthy_routing_fraction",
        "load_balancer_error_rate_percent_max": "load_balancer_error_rate_percent",
        "workload_storage_utilization_ratio_max": "workload_storage_utilization_ratio",
        "workload_storage_latency_penalty_ms_max": "workload_storage_latency_penalty_ms",
        "workload_application_error_rate_percent_max": "workload_application_error_rate_percent",
        "workload_dropped_requests_per_second_max": "workload_dropped_requests_per_second",
        "workload_current_demand_per_second_max": "workload_current_demand_per_second",
        "power_budget_violating_racks_max": "power_budget_violating_racks",
        "max_power_budget_utilization_ratio_max": "max_power_budget_utilization_ratio",
        "temperature_sensor_unhealthy_count_max": "temperature_sensor_unhealthy_count",
        "max_temperature_sensor_disagreement_c_max": "max_temperature_sensor_disagreement_c",
        "host_health_flapping_count_max": "host_health_flapping_count",
        "thermal_throttled_servers_max": "thermal_throttled_servers",
    }
    for criterion_name, summary_field in threshold_fields.items():
        if (
            criterion_name in criteria
            and summary.get(summary_field, float("inf")) > criteria[criterion_name]
        ):
            return False
    min_fields = {
        "min_thermal_throttle_factor_min": "min_thermal_throttle_factor",
        "workload_service_capacity_requests_per_second_min": "workload_service_capacity_requests_per_second",
        "autoscaler_effective_server_limit_min": "autoscaler_effective_server_limit",
    }
    for criterion_name, summary_field in min_fields.items():
        if (
            criterion_name in criteria
            and summary.get(summary_field, float("-inf")) < criteria[criterion_name]
        ):
            return False
    avoided_racks = criteria.get("workload_allocated_away_from_rack")
    if avoided_racks is not None:
        if isinstance(avoided_racks, str):
            avoided_racks = [avoided_racks]
        allocated_server_ids = summary.get("workload_allocated_server_ids") or []
        if (
            not allocated_server_ids
            and summary.get("workload_current_demand_per_second", 0.0) > 0.0
        ):
            return False
        for rack_id in avoided_racks:
            if server_ids_include_rack(allocated_server_ids, rack_id):
                return False
    avoided_servers = criteria.get("workload_allocated_away_from_server")
    if avoided_servers is not None:
        if isinstance(avoided_servers, str):
            avoided_servers = [avoided_servers]
        allocated_server_ids = set(summary.get("workload_allocated_server_ids") or [])
        if (
            not allocated_server_ids
            and summary.get("workload_current_demand_per_second", 0.0) > 0.0
        ):
            return False
        if any(server_id in allocated_server_ids for server_id in avoided_servers):
            return False
    required_rack = criteria.get("workload_allocated_to_rack")
    if required_rack is not None and not server_ids_include_rack(
        summary.get("workload_allocated_server_ids") or [],
        required_rack,
    ):
        return False
    return True


def declared_mitigation_actions(scenario):
    actions = scenario.success_criteria.get("mitigation_actions")
    if actions is None:
        actions = [scenario.success_criteria["mitigation_action"]]
    return actions


def agent_action_matches_declared_mitigation(call, scenario):
    request = call.get("request") or {}
    response = call.get("response") or {}
    if not isinstance(request, dict) or not isinstance(response, dict):
        return False
    if not response.get("accepted"):
        return False
    if not action_started_during_scenario_fault(response, scenario):
        return False

    request_parameters = effective_control_request_parameters(request)
    for action in declared_mitigation_actions(scenario):
        if request.get("action_type") != action.get("action_type"):
            continue
        if request_parameters == dict(action.get("parameters", {})):
            return True
    return False


def action_started_during_scenario_fault(response, scenario):
    action_start = response.get("sim_time_seconds_before")
    if not isinstance(action_start, (int, float)):
        return False
    for fault in response.get("active_faults_before") or []:
        if (
            fault.get("fault_type") != scenario.fault_type
            or fault.get("target") != scenario.fault_target
        ):
            continue
        return (
            action_start
            < fault["started_at_sim_time_seconds"] + fault["duration_seconds"]
        )
    return False


def effective_control_request_parameters(request):
    parameters = request.get("parameters") or {}
    merged = dict(parameters) if isinstance(parameters, dict) else {}
    for key in (
        "target",
        "fan_speed_percent",
        "supply_air_temperature_c",
        "source_rack_id",
        "target_rack_id",
        "tenant_id",
        "workload_fraction",
        "request_rate_per_second",
        "server_id",
        "calibration_offset_c",
        "mark_untrusted",
        "min_capacity",
        "max_capacity",
        "target_utilization_percent",
        "cooldown_seconds",
        "placement_strategy",
        "forbidden_rack_ids",
        "max_server_count",
        "backend_weights",
        "remove_backend_ids",
        "add_backend_ids",
        "routing_policy",
        "reset_to_equal_weights",
    ):
        value = request.get(key)
        if value is not None:
            merged[key] = value
    return merged


def apply_tracked_agent_action(sim, action_type, parameters=None, advance_ticks=1):
    request = {
        "action_type": action_type,
        "parameters": dict(parameters or {}),
        "advance_ticks": int(advance_ticks),
        "include_config": True,
        "log_limit": 20,
    }
    response = apply_agent_action(
        sim, AgentActionRequest(**request), visibility="evaluator"
    )
    sim._benchmark_agent_actions = getattr(sim, "_benchmark_agent_actions", []) + [
        {"request": request, "response": response}
    ]
    return response


def apply_manifest_mitigation(sim, scenario):
    response = None
    for action in declared_mitigation_actions(scenario):
        response = apply_tracked_agent_action(
            sim,
            action["action_type"],
            parameters=action.get("parameters", {}),
            advance_ticks=action.get("advance_ticks", 1),
        )
        assert response["accepted"] is True
    summary = response["observation"]["summary"] if response else sim.state_summary()
    stability_window = int(scenario.success_criteria.get("stability_window_seconds", 0))
    if stability_window:
        summary = sim.step(ticks=stability_window)
    return summary


def wrong_same_type_mitigation_action(scenario):
    action = declared_mitigation_actions(scenario)[0]
    action_type = action["action_type"]
    parameters = dict(action.get("parameters", {}))
    if action_type == "set_cooling":
        parameters["fan_speed_percent"] = 0
        parameters["supply_air_temperature_c"] = 40
    elif action_type == "calibrate_sensor":
        parameters["calibration_offset_c"] = 0
        parameters.pop("mark_untrusted", None)
    elif action_type == "migrate_workload":
        parameters["workload_fraction"] = 0.5
    elif action_type == "throttle_workload":
        parameters["request_rate_per_second"] = (
            parameters["request_rate_per_second"] + 100
        )
    elif action_type == "update_autoscaler_policy":
        parameters["max_capacity"] = max(1, int(parameters.get("max_capacity", 1)) - 1)
    elif action_type == "repair_monitoring_pipeline":
        parameters["target"] = "standby-monitoring-pipeline"
    elif action_type == "update_placement_policy":
        parameters["forbidden_rack_ids"] = []
        parameters["max_server_count"] = 1
    elif action_type == "update_load_balancer_config":
        parameters["reset_to_equal_weights"] = False
        parameters["routing_policy"] = "sticky"
        parameters.pop("remove_backend_ids", None)
    elif action_type == "set_server_maintenance":
        parameters.pop("server_id", None)
        parameters["rack_id"] = "rack-r1-row1-01"
    return {
        "action_type": action_type,
        "parameters": parameters,
        "advance_ticks": int(action.get("advance_ticks", 1)),
    }


def server_ids_include_rack(server_ids, rack_id):
    parts = rack_id.split("-")
    if len(parts) != 4 or parts[0] != "rack":
        return False
    server_prefix = f"server-{parts[1]}-{parts[2]}-rack{parts[3]}-"
    return any(
        isinstance(server_id, str) and server_id.startswith(server_prefix)
        for server_id in server_ids
    )


@pytest.mark.parametrize(
    "scenario",
    BENCHMARK_SCENARIOS,
    ids=[scenario.problem_id for scenario in BENCHMARK_SCENARIOS],
)
def test_registered_benchmark_scenarios_execute_against_real_simulator(scenario):
    sim, before_fault, post_fault, observation = run_benchmark_scenario(scenario)

    active_faults = observation["summary"]["active_faults"]
    observation_text = json.dumps(observation, sort_keys=True, default=str).lower()
    changed_fields = {
        key
        for key, value in post_fault.items()
        if before_fault.get(key) != value
        and key not in {"sim_time_seconds", "active_faults"}
    }

    assert sim.config.simulation.auto_advance is False
    assert any(
        fault["fault_type"] == scenario.fault_type
        and fault["target"] == scenario.fault_target
        for fault in active_faults
    )
    assert changed_fields
    for evidence_term in scenario.expected["evidence_terms"]:
        assert evidence_term.lower() in observation_text


def test_agent_visible_action_schema_passes_leakage_gate():
    schema = action_space()

    assert "domain_contract" not in schema
    assert "benchmark_action_coverage" not in schema
    assert_no_agent_leakage(schema)


@pytest.mark.parametrize(
    "scenario",
    BENCHMARK_SCENARIOS,
    ids=[scenario.problem_id for scenario in BENCHMARK_SCENARIOS],
)
def test_registered_scenario_agent_visible_observations_and_action_responses_pass_leakage_gate(
    scenario,
):
    sim = make_benchmark_simulator(scenario)
    before_observation = agent_observation(sim, log_limit=20, include_config=True)

    inject_scenario_fault(sim, scenario)
    sim.step(ticks=1)
    after_observation = agent_observation(sim, log_limit=20, include_config=True)
    observe_response = apply_agent_action(
        sim,
        AgentActionRequest(action_type="observe", log_limit=20, include_config=True),
    )
    noop_response = apply_agent_action(
        sim,
        AgentActionRequest(
            action_type="noop", advance_ticks=1, log_limit=20, include_config=True
        ),
    )
    step_response = apply_agent_action(
        sim,
        AgentActionRequest(
            action_type="step",
            parameters={"ticks": 1},
            log_limit=20,
            include_config=True,
        ),
    )

    assert_no_agent_leakage(before_observation)
    assert_no_agent_leakage(after_observation)
    assert_no_agent_leakage(observe_response)
    assert_no_agent_leakage(noop_response)
    assert_no_agent_leakage(step_response)

    if scenario.task_type == "mitigation":
        for action in declared_mitigation_actions(scenario):
            response = apply_agent_action(
                sim,
                AgentActionRequest(
                    action_type=action["action_type"],
                    parameters=dict(action.get("parameters", {})),
                    advance_ticks=int(action.get("advance_ticks", 1)),
                    log_limit=20,
                    include_config=True,
                ),
            )
            assert response["accepted"] is True
            assert_no_agent_leakage(response)


@pytest.mark.parametrize(
    "scenario",
    REPRESENTATIVE_SCENARIOS_BY_FAULT,
    ids=[scenario.fault_type for scenario in REPRESENTATIVE_SCENARIOS_BY_FAULT],
)
def test_each_fault_family_changes_agent_visible_symptom_after_injection(scenario):
    sim = make_benchmark_simulator(scenario)
    before_observation = agent_observation(sim, log_limit=20, include_config=True)

    inject_scenario_fault(sim, scenario)
    sim.step(ticks=1)
    after_observation = agent_observation(sim, log_limit=20, include_config=True)
    changed_fields = changed_agent_visible_symptom_fields(
        before_observation, after_observation
    )

    assert_no_agent_leakage(before_observation)
    assert_no_agent_leakage(after_observation)
    assert changed_fields


@pytest.mark.parametrize(
    "scenario",
    REPRESENTATIVE_SCENARIOS_BY_FAULT,
    ids=[scenario.fault_type for scenario in REPRESENTATIVE_SCENARIOS_BY_FAULT],
)
def test_each_fault_family_exposes_safe_agent_visible_alert_or_metric_signal(scenario):
    sim = make_benchmark_simulator(scenario)
    before_observation = agent_observation(sim, log_limit=20, include_config=True)

    inject_scenario_fault(sim, scenario)
    sim.step(ticks=1)
    after_observation = agent_observation(sim, log_limit=20, include_config=True)
    changed_fields = changed_agent_visible_symptom_fields(
        before_observation, after_observation
    )
    safe_signals = safe_signal_fields_for_fault(scenario.fault_type, changed_fields)

    assert_no_agent_leakage(after_observation)
    assert safe_signals


@pytest.mark.parametrize(
    "scenario",
    [
        scenario
        for scenario in BENCHMARK_SCENARIOS
        if scenario.task_type == "mitigation"
    ],
    ids=[
        scenario.problem_id
        for scenario in BENCHMARK_SCENARIOS
        if scenario.task_type == "mitigation"
    ],
)
def test_registered_mitigation_scenarios_fail_noop_and_pass_declared_action(scenario):
    noop_sim, _before_fault, _post_fault, _observation = run_benchmark_scenario(
        scenario
    )
    noop_response = apply_agent_action(
        noop_sim,
        AgentActionRequest(
            action_type="noop",
            advance_ticks=int(scenario.success_criteria["stability_window_seconds"]),
            include_config=True,
            log_limit=20,
        ),
        visibility="evaluator",
    )
    noop_summary = noop_response["observation"]["summary"]

    action_sim, _before_fault, _post_fault, _observation = run_benchmark_scenario(
        scenario
    )
    action_summary = apply_manifest_mitigation(action_sim, scenario)

    assert noop_response["accepted"] is True
    assert summary_satisfies_success_criteria(noop_summary, scenario, noop_sim) is False
    assert (
        summary_satisfies_success_criteria(action_summary, scenario, action_sim) is True
    )


@pytest.mark.parametrize(
    "scenario",
    [
        scenario
        for scenario in BENCHMARK_SCENARIOS
        if scenario.task_type == "mitigation"
    ],
    ids=[
        scenario.problem_id
        for scenario in BENCHMARK_SCENARIOS
        if scenario.task_type == "mitigation"
    ],
)
def test_registered_mitigation_scenarios_do_not_pass_by_waiting_for_fault_expiry(
    scenario,
):
    sim, _before_fault, _post_fault, _observation = run_benchmark_scenario(scenario)
    response = apply_agent_action(
        sim,
        AgentActionRequest(
            action_type="noop",
            advance_ticks=scenario.fault_duration_seconds,
            include_config=True,
            log_limit=20,
        ),
        visibility="evaluator",
    )
    summary = response["observation"]["summary"]
    wait_only_summary = summary
    summary = apply_manifest_mitigation(sim, scenario)

    assert response["accepted"] is True
    assert response["active_faults_after"] == []
    assert summary_satisfies_success_criteria(wait_only_summary, scenario, sim) is False
    assert summary["active_faults"] == []
    assert summary_satisfies_success_criteria(summary, scenario, sim) is False


@pytest.mark.parametrize(
    "scenario",
    [
        scenario
        for scenario in BENCHMARK_SCENARIOS
        if scenario.task_type == "mitigation"
    ],
    ids=[
        scenario.problem_id
        for scenario in BENCHMARK_SCENARIOS
        if scenario.task_type == "mitigation"
    ],
)
def test_registered_mitigation_scenarios_do_not_credit_wrong_same_type_action(scenario):
    sim, _before_fault, _post_fault, _observation = run_benchmark_scenario(scenario)
    wrong_action = wrong_same_type_mitigation_action(scenario)
    action_response = apply_tracked_agent_action(
        sim,
        wrong_action["action_type"],
        parameters=wrong_action["parameters"],
        advance_ticks=wrong_action["advance_ticks"],
    )
    expiry_response = apply_agent_action(
        sim,
        AgentActionRequest(
            action_type="noop",
            advance_ticks=scenario.fault_duration_seconds,
            include_config=True,
            log_limit=20,
        ),
        visibility="evaluator",
    )
    summary = expiry_response["observation"]["summary"]
    stability_window = int(scenario.success_criteria.get("stability_window_seconds", 0))
    if stability_window:
        summary = sim.step(ticks=stability_window)

    assert action_response["accepted"] is True
    assert action_response["active_faults_before"]
    assert expiry_response["active_faults_after"] == []
    assert summary_satisfies_success_criteria(summary, scenario, sim) is False


def test_agent_contract_operational_domains_have_mitigation_scenarios():
    incident_domains = action_space(visibility="evaluator")["domain_contract"][
        "incident_domains"
    ]
    faults_requiring_operations = {
        fault_type
        for fault_type, details in incident_domains.items()
        if set(details["agent_response_actions"]) - {"observe"}
    }
    mitigation_faults = {
        scenario.fault_type
        for scenario in BENCHMARK_SCENARIOS
        if scenario.task_type == "mitigation"
    }

    assert faults_requiring_operations <= mitigation_faults


def test_agent_contract_write_actions_have_scored_or_documented_benchmark_status():
    contract = action_space(visibility="evaluator")
    coverage = contract["benchmark_action_coverage"]
    scored_actions = set(coverage["scored_actions"])
    non_scored_actions = set(coverage["supported_non_scored_actions"])
    manifest_mitigation_actions = {
        action["action_type"]
        for scenario in BENCHMARK_SCENARIOS
        if scenario.task_type == "mitigation"
        for action in declared_mitigation_actions(scenario)
    }

    assert set(contract["control_actions"]) == SUPPORTED_ACTIONS
    assert scored_actions == manifest_mitigation_actions
    assert SUPPORTED_ACTIONS <= scored_actions | non_scored_actions
    assert coverage["all_write_actions_have_benchmark_status"] is True
    assert (
        contract["domain_contract"]["coverage_invariant"][
            "all_write_actions_have_benchmark_status"
        ]
        is True
    )


def test_agent_api_supports_documented_non_scored_maintenance_actions():
    sim = make_sim()
    server_id = "server-r1-row1-rack01-01"

    set_response = apply_agent_action(
        sim,
        AgentActionRequest(
            action_type="set_server_maintenance",
            parameters={"server_id": server_id},
            advance_ticks=0,
            include_config=True,
        ),
    )
    assert set_response["accepted"] is True
    assert sim._require_server(server_id).status == "maintenance"

    clear_response = apply_agent_action(
        sim,
        AgentActionRequest(
            action_type="clear_server_maintenance",
            parameters={"server_id": server_id},
            advance_ticks=0,
            include_config=True,
        ),
    )

    assert clear_response["accepted"] is True
    assert sim._require_server(server_id).status == "healthy"


def test_agent_api_supports_rack_scoped_maintenance_without_host_disclosure():
    sim = make_sim()
    rack_id = "rack-r1-row1-01"
    rack = sim._require_rack(rack_id)

    set_response = apply_agent_action(
        sim,
        AgentActionRequest(
            action_type="set_server_maintenance",
            parameters={"rack_id": rack_id},
            advance_ticks=0,
            include_config=True,
        ),
    )

    assert set_response["accepted"] is True
    assert all(server.status == "maintenance" for server in rack.servers)
    assert "server-" not in json.dumps(
        set_response,
        sort_keys=True,
        default=str,
    )

    clear_response = apply_agent_action(
        sim,
        AgentActionRequest(
            action_type="clear_server_maintenance",
            parameters={"rack_id": rack_id},
            advance_ticks=0,
            include_config=True,
        ),
    )

    assert clear_response["accepted"] is True
    assert all(server.status == "healthy" for server in rack.servers)


def test_reset_determinism():
    sim_a = make_sim()
    sim_b = make_sim()
    workload = {"request_rate_per_second": 300, "placement_strategy": "random"}

    sim_a.reset(seed=7)
    sim_a.start_workload(workload)
    for _ in range(5):
        sim_a.step()

    sim_b.reset(seed=7)
    sim_b.start_workload(workload)
    for _ in range(5):
        sim_b.step()

    assert sim_a.state_summary() == sim_b.state_summary()


def test_step_progression():
    sim = make_sim()
    sim.step({"ticks": 1}["ticks"])
    assert sim.state_summary()["sim_time_seconds"] == 1


def test_server_power_calculation():
    sim = make_sim()
    sim.start_workload(
        {
            "request_rate_per_second": 800,
            "placement_strategy": "rack_hotspot",
            "target_rack_id": "rack-r1-row1-01",
        }
    )
    server = sim._require_server("server-r1-row1-rack01-01")
    expected = (
        sim.config.server.idle_power_kw
        + server.cpu_utilization_percent / 100.0 * sim.config.server.dynamic_power_kw
    )
    assert round(server.power_kw, 6) == round(expected, 6)


def test_rack_temperature_calculation_changes_with_hotspot():
    sim = make_sim()
    rack = sim._require_rack("rack-r1-row1-02")
    baseline = rack.inlet_temperature_c
    sim.inject_fault(
        FaultRequest(
            fault_type="rack_hotspot",
            target=rack.rack_id,
            severity=0.4,
            duration_seconds=60,
        )
    )
    assert rack.inlet_temperature_c > baseline
    assert rack.thermal_status in {"warning", "critical"}


def test_cooling_degradation_fault_effect():
    sim = make_sim()
    baseline_capacity = sim.cooling_units[0].cooling_capacity_kw
    baseline_temp = sim._require_rack("rack-r1-row1-01").inlet_temperature_c
    sim.inject_fault(
        FaultRequest(
            fault_type="cooling_degradation",
            target="cooling-unit-1",
            severity=0.5,
            duration_seconds=300,
        )
    )
    assert sim.cooling_units[0].cooling_capacity_kw < baseline_capacity
    assert sim._require_rack("rack-r1-row1-01").inlet_temperature_c > baseline_temp


def test_server_failure_fault_effect():
    sim = make_sim()
    server_id = "server-r1-row1-rack02-07"
    sim.inject_fault(
        FaultRequest(
            fault_type="server_failure",
            target=server_id,
            severity=1.0,
            duration_seconds=60,
        )
    )
    assert sim._require_server(server_id).status == "failed"
    assert sim.state_summary()["failed_servers"] == 1


def test_control_action_application():
    sim = make_sim()
    result = sim.apply_control(
        ControlRequest(
            action_type="set_cooling",
            target="cooling-unit-1",
            fan_speed_percent=90,
            supply_air_temperature_c=18,
        )
    )
    assert result["status"] == "applied"
    assert sim.cooling_units[0].fan_speed_percent == 90
    assert sim.cooling_units[0].supply_air_temperature_c == 18


def test_sla_violation_logic_for_power_overload():
    sim = make_sim()
    sim.inject_fault(
        FaultRequest(
            fault_type="power_overload",
            target="rack-r1-row2-03",
            severity=0.3,
            duration_seconds=300,
        )
    )
    assert sim._require_rack("rack-r1-row2-03").power_status == "overloaded"
    assert sim.state_summary()["sla_status"] == "violated"


def test_network_partition_fault_makes_rack_unreachable_without_failing_servers():
    sim = make_sim()
    rack_id = "rack-r1-row1-01"
    server_id = "server-r1-row1-rack01-01"
    sim.reset(seed=7)
    sim.start_workload(
        {
            "request_rate_per_second": 1000,
            "placement_strategy": "rack_hotspot",
            "target_rack_id": rack_id,
            "noise_enabled": False,
        }
    )
    before = sim.step(ticks=1)

    sim.inject_fault(
        FaultRequest(
            fault_type="network_partition",
            target=rack_id,
            severity=1.0,
            duration_seconds=30,
        )
    )
    after = sim.step(ticks=1)

    assert before["workload_allocated_server_ids"]
    assert after["workload_allocated_server_ids"] == []
    assert after["workload_desired_allocation_weights"]
    assert after["workload_queue_length"] > before["workload_queue_length"]
    assert sim._require_server(server_id).status == "healthy"


def test_network_partition_zero_severity_is_noop_for_single_workload():
    workload = {
        "request_rate_per_second": 1000,
        "placement_strategy": "rack_hotspot",
        "target_rack_id": "rack-r1-row1-01",
        "noise_enabled": False,
    }
    baseline = make_sim()
    baseline.reset(seed=7)
    baseline.start_workload(workload)
    baseline.step(ticks=1)
    expected = baseline.step(ticks=1)

    sim = make_sim()
    sim.reset(seed=7)
    sim.start_workload(workload)
    sim.step(ticks=1)
    sim.inject_fault(
        FaultRequest(
            fault_type="network_partition",
            target="rack-r1-row1-01",
            severity=0.0,
            duration_seconds=30,
        )
    )
    observed = sim.step(ticks=1)

    assert (
        observed["workload_allocated_server_ids"]
        == expected["workload_allocated_server_ids"]
    )
    assert (
        observed["workload_allocation_weights"]
        == expected["workload_allocation_weights"]
    )
    assert observed["workload_queue_length"] == expected["workload_queue_length"]
    assert (
        observed["workload_service_capacity_requests_per_second"]
        == expected["workload_service_capacity_requests_per_second"]
    )
    assert (
        observed["workload_average_latency_ms"]
        == expected["workload_average_latency_ms"]
    )


def test_network_partition_partial_severity_reduces_capacity_without_disconnect():
    sim = make_sim()
    sim.reset(seed=7)
    sim.start_workload(
        {
            "request_rate_per_second": 10000,
            "placement_strategy": "rack_hotspot",
            "target_rack_id": "rack-r1-row1-01",
            "noise_enabled": False,
        }
    )
    baseline = sim.step(ticks=1)

    sim.inject_fault(
        FaultRequest(
            fault_type="network_partition",
            target="rack-r1-row1-01",
            severity=0.5,
            duration_seconds=30,
        )
    )
    degraded = sim.step(ticks=1)

    assert (
        degraded["workload_allocated_server_ids"]
        == baseline["workload_allocated_server_ids"]
    )
    assert sum(degraded["workload_allocation_weights"].values()) == pytest.approx(
        sum(baseline["workload_allocation_weights"].values()) * 0.5
    )
    assert degraded["workload_service_capacity_requests_per_second"] == pytest.approx(
        baseline["workload_service_capacity_requests_per_second"] * 0.5
    )
    assert degraded["workload_queue_length"] > baseline["workload_queue_length"]


def test_network_congestion_burst_only_amplifies_during_burst_window():
    sim = make_sim()
    sim.reset(seed=7)
    sim.start_workload(
        {
            "request_rate_per_second": 1000,
            "workload_class": "network_heavy",
            "workload_profile_type": "burst",
            "workload_profile_parameters": {
                "baseline_rate_per_second": 1000,
                "burst_rate_per_second": 16000,
                "burst_start_time_seconds": 2,
                "burst_duration_seconds": 2,
            },
            "network_capacity_mbps": 300,
            "placement_strategy": "rack_hotspot",
            "target_rack_id": "rack-r1-row1-03",
            "noise_enabled": False,
        }
    )
    sim.inject_fault(
        FaultRequest(
            fault_type="network_congestion_burst",
            target="workload",
            severity=0.9,
            duration_seconds=30,
        )
    )

    before_burst = sim.step(ticks=1)
    during_burst = sim.step(ticks=1)
    burst_alert_types = {alert["alert_type"] for alert in sim.alerts()}
    after_burst = sim.step(ticks=3)

    assert before_burst["workload_network_congestion_ratio"] < 0.7
    assert before_burst["workload_queue_length"] == 0
    assert during_burst["workload_network_congestion_ratio"] > 1.0
    assert (
        during_burst["workload_average_latency_ms"]
        > before_burst["workload_average_latency_ms"]
    )
    assert during_burst["workload_queue_length"] > before_burst["workload_queue_length"]
    assert (
        after_burst["workload_network_congestion_ratio"]
        < during_burst["workload_network_congestion_ratio"]
    )
    assert "NetworkCongestionElevated" in burst_alert_types


def test_tor_packet_loss_increases_loss_and_latency_without_failing_servers():
    sim = make_sim()
    target = "rack-r1-row1-03"
    sim.reset(seed=7)
    sim.start_workload(
        {
            "request_rate_per_second": 2500,
            "workload_class": "network_heavy",
            "network_capacity_mbps": 1000,
            "placement_strategy": "rack_hotspot",
            "target_rack_id": target,
            "noise_enabled": False,
        }
    )
    baseline = sim.step(ticks=1)

    sim.inject_fault(
        FaultRequest(
            fault_type="tor_packet_loss",
            target=target,
            severity=0.85,
            duration_seconds=30,
        )
    )
    degraded = sim.step(ticks=1)
    rack = sim._require_rack(target)

    assert (
        degraded["workload_allocated_server_ids"]
        == baseline["workload_allocated_server_ids"]
    )
    assert degraded["workload_network_packet_loss_percent"] > 0.0
    assert degraded["workload_network_retransmit_rate"] > 0.0
    assert degraded["workload_network_error_rate"] > 0.0
    assert (
        degraded["workload_average_latency_ms"]
        > baseline["workload_average_latency_ms"]
    )
    assert degraded["workload_network_affected_rack_id"] == target
    assert rack.network_path_status == "degraded"
    assert degraded["failed_servers"] == 0
    assert all(server.status == "healthy" for server in rack.servers)
    assert any(alert["alert_type"] == "PacketLossElevated" for alert in sim.alerts())


def test_network_partition_zero_severity_is_noop_for_tenant_workload():
    tenants = [
        {
            "tenant_id": "tenant-a",
            "request_rate_per_second": 500,
            "placement_strategy": "rack_hotspot",
            "target_rack_id": "rack-r1-row1-01",
        }
    ]
    baseline = tenant_sim(tenants)
    baseline.step(ticks=1)
    expected = baseline.step(ticks=1)["tenant_summaries"]["tenant-a"]

    sim = tenant_sim(tenants)
    sim.step(ticks=1)
    sim.inject_fault(
        FaultRequest(
            fault_type="network_partition",
            target="rack-r1-row1-01",
            severity=0.0,
            duration_seconds=30,
        )
    )
    observed = sim.step(ticks=1)["tenant_summaries"]["tenant-a"]

    assert observed["allocated_server_ids"] == expected["allocated_server_ids"]
    assert observed["allocation_weights"] == expected["allocation_weights"]
    assert observed["queue_length"] == expected["queue_length"]
    assert (
        observed["service_capacity_requests_per_second"]
        == expected["service_capacity_requests_per_second"]
    )
    assert observed["average_latency_ms"] == expected["average_latency_ms"]


def test_storage_io_saturation_fault_increases_storage_latency():
    sim = make_sim()
    sim.reset(seed=7)
    sim.start_workload(
        {
            "request_rate_per_second": 2000,
            "workload_class": "storage_heavy",
            "placement_strategy": "spread",
            "noise_enabled": False,
        }
    )
    baseline = sim.step(ticks=1)

    sim.inject_fault(
        FaultRequest(
            fault_type="storage_io_saturation",
            target="storage",
            severity=0.9,
            duration_seconds=30,
        )
    )
    degraded = sim.step(ticks=1)

    assert (
        degraded["workload_storage_utilization_ratio"]
        > baseline["workload_storage_utilization_ratio"]
    )
    assert (
        degraded["workload_storage_latency_penalty_ms"]
        > baseline["workload_storage_latency_penalty_ms"]
    )


def test_control_plane_degradation_fault_reduces_service_capacity():
    config = default_config().merged(
        {
            "simulation": {"auto_advance": False},
            "topology": {
                "rooms": 1,
                "rows_per_room": 1,
                "racks_per_row": 1,
                "servers_per_rack": 2,
            },
        }
    )
    sim = DataCenterSimulator(config)
    sim.reset(seed=7)
    sim.start_workload(
        {
            "request_rate_per_second": 1000,
            "placement_strategy": "spread",
            "noise_enabled": False,
        }
    )
    baseline = sim.step(ticks=1)

    sim.inject_fault(
        FaultRequest(
            fault_type="control_plane_degradation",
            target="control-plane",
            severity=0.9,
            duration_seconds=30,
        )
    )
    degraded = sim.step(ticks=1)

    assert (
        degraded["workload_service_capacity_requests_per_second"]
        < baseline["workload_service_capacity_requests_per_second"]
    )
    assert degraded["workload_queue_length"] > baseline["workload_queue_length"]
    assert (
        degraded["workload_average_latency_ms"]
        > baseline["workload_average_latency_ms"]
    )


def test_autoscaler_misconfiguration_limits_capacity_and_policy_update_recovers():
    sim = make_sim()
    sim.reset(seed=7)
    sim.start_workload(
        {
            "request_rate_per_second": 12000,
            "placement_strategy": "spread",
            "noise_enabled": False,
        }
    )
    baseline = sim.step(ticks=1)

    sim.inject_fault(
        FaultRequest(
            fault_type="autoscaler_misconfiguration",
            target="autoscaler",
            severity=0.9,
            duration_seconds=300,
        )
    )
    degraded = sim.step(ticks=1)

    assert baseline["workload_service_capacity_requests_per_second"] >= 12000
    assert degraded["autoscaler_status"] == "misconfigured"
    assert degraded["autoscaler_effective_server_limit"] < 10
    assert (
        degraded["workload_service_capacity_requests_per_second"]
        < degraded["workload_current_demand_per_second"]
    )
    assert degraded["workload_queue_length"] > 0
    assert any(
        alert["alert_type"] == "AutoscalerPolicyLimited" for alert in sim.alerts()
    )

    sim.apply_control(
        ControlRequest(
            action_type="update_autoscaler_policy",
            min_capacity=4,
            max_capacity=80,
            target_utilization_percent=65,
            cooldown_seconds=30,
        )
    )
    recovered = sim.step(ticks=10)

    assert recovered["autoscaler_status"] == "updated"
    assert recovered["autoscaler_effective_server_limit"] >= 40
    assert recovered["workload_queue_length"] == 0
    assert recovered["sla_status"] == "normal"


def test_monitoring_pipeline_failure_stales_agent_telemetry_but_not_evaluator_state():
    from dc_twin.agent_interface import agent_observation, get_evaluator_state

    sim = make_sim()
    sim.reset(seed=7)
    sim.start_workload(
        {
            "request_rate_per_second": 300,
            "placement_strategy": "spread",
            "noise_enabled": False,
        }
    )
    sim.step(ticks=5)

    sim.inject_fault(
        FaultRequest(
            fault_type="monitoring_pipeline_failure",
            target="monitoring-pipeline",
            severity=0.85,
            duration_seconds=300,
        )
    )
    degraded = sim.step(ticks=3)
    agent_view = agent_observation(sim, log_limit=20, include_config=True)
    evaluator_state = get_evaluator_state(sim, log_limit=20, include_config=True)

    assert degraded["telemetry_lag_seconds"] > 0
    assert degraded["metrics_missing_ratio"] > 0.0
    assert degraded["logs_missing_ratio"] > 0.0
    assert degraded["sla_status"] == "normal"
    assert any(alert["alert_type"] == "TelemetryStale" for alert in sim.alerts())
    assert agent_view["summary"]["telemetry_lag_seconds"] > 0
    assert agent_view["summary"]["workload_average_latency_ms"] is None
    assert agent_view["summary"]["average_rack_inlet_temperature_c"] is None
    assert agent_view["sla_status"] == "unknown"
    assert {alert["alert_type"] for alert in agent_view["alerts"]} == {"TelemetryStale"}
    assert all(
        event.get("sim_time_seconds", 0)
        <= agent_view["summary"]["logs_last_updated_sim_time_seconds"]
        for event in agent_view["recent_events"]
    )
    assert "workload_average_latency_ms" in evaluator_state["summary"]
    assert (
        evaluator_state["active_faults"][0]["fault_type"]
        == "monitoring_pipeline_failure"
    )
    assert (
        evaluator_state["summary"]["sim_time_seconds"]
        == sim.state_summary()["sim_time_seconds"]
    )

    sim.apply_control(
        ControlRequest(
            action_type="repair_monitoring_pipeline", target="monitoring-pipeline"
        )
    )
    recovered = sim.step(ticks=1)

    assert recovered["telemetry_lag_seconds"] == 0
    assert recovered["metrics_missing_ratio"] == 0.0
    assert recovered["logs_missing_ratio"] == 0.0


def test_placement_policy_misconfiguration_forces_bad_rack_until_policy_update():
    sim = make_sim()
    target = "rack-r1-row1-04"
    sim.reset(seed=7)
    sim.start_workload(
        {
            "request_rate_per_second": 8000,
            "placement_strategy": "spread",
            "noise_enabled": False,
        }
    )
    baseline = sim.step(ticks=1)

    sim.inject_fault(
        FaultRequest(
            fault_type="placement_policy_misconfiguration",
            target=target,
            severity=0.9,
            duration_seconds=300,
        )
    )
    degraded = sim.step(ticks=1)

    assert baseline["placement_policy_violating_racks"] == 0
    assert degraded["placement_policy_status"] == "misconfigured"
    assert degraded["placement_policy_violating_racks"] == 1
    assert server_ids_include_rack(degraded["workload_allocated_server_ids"], target)
    assert degraded["workload_placement_imbalance_ratio"] == 1.0
    assert degraded["workload_queue_length"] > 0
    assert any(alert["alert_type"] == "PlacementPolicyDrift" for alert in sim.alerts())

    sim.apply_control(
        ControlRequest(
            action_type="update_placement_policy",
            placement_strategy="spread",
            forbidden_rack_ids=[target],
            max_server_count=40,
        )
    )
    recovered = sim.step(ticks=10)

    assert recovered["placement_policy_status"] == "updated"
    assert recovered["placement_policy_violating_racks"] == 0
    assert not server_ids_include_rack(
        recovered["workload_allocated_server_ids"], target
    )
    assert recovered["workload_queue_length"] == 0
    assert recovered["sla_status"] == "normal"


def test_load_balancer_misconfiguration_skews_backend_routing_and_config_update_recovers():
    sim = make_sim()
    unhealthy_backend_id = "server-r1-row1-rack02-03"
    sim.reset(seed=7)
    sim.start_workload(
        {
            "request_rate_per_second": 4500,
            "placement_strategy": "spread",
            "noise_enabled": False,
        }
    )
    baseline = sim.step(ticks=1)

    injected = sim.inject_fault(
        FaultRequest(
            fault_type="load_balancer_misconfiguration",
            target="load-balancer",
            severity=0.9,
            duration_seconds=300,
        )
    )
    degraded = sim.step(ticks=1)
    alert_types = {alert["alert_type"] for alert in sim.alerts()}

    assert baseline["load_balancer_error_rate_percent"] == 0.0
    assert degraded["load_balancer_backend_skew_ratio"] > 0.35
    assert degraded["load_balancer_unhealthy_routing_fraction"] > 0.0
    assert degraded["load_balancer_error_rate_percent"] > 0.0
    assert (
        degraded["workload_application_error_rate_percent"]
        == degraded["load_balancer_error_rate_percent"]
    )
    assert degraded["workload_dropped_requests_per_second"] > 0.0
    assert degraded["workload_queue_length"] > baseline["workload_queue_length"]
    assert unhealthy_backend_id in degraded["load_balancer_unhealthy_backend_ids"]
    assert degraded["failed_servers"] == 0
    assert all(server.status == "healthy" for server in sim.servers)
    assert {
        "LoadBalancerBackendImbalance",
        "LoadBalancerRoutingUnhealthyBackend",
    } <= alert_types

    sim.apply_control(
        ControlRequest(
            action_type="update_load_balancer_config",
            remove_backend_ids=[unhealthy_backend_id],
            routing_policy="round_robin",
            reset_to_equal_weights=True,
        )
    )
    recovered = sim.step(ticks=10)

    assert recovered["load_balancer_routing_policy"] == "round_robin"
    assert unhealthy_backend_id not in recovered["load_balancer_backend_server_ids"]
    assert recovered["load_balancer_backend_skew_ratio"] <= 0.1
    assert recovered["load_balancer_unhealthy_routing_fraction"] == 0.0
    assert recovered["load_balancer_error_rate_percent"] == 0.0
    assert recovered["workload_application_error_rate_percent"] == 0.0
    assert recovered["workload_dropped_requests_per_second"] == 0.0
    assert recovered["workload_queue_length"] == 0
    assert recovered["sla_status"] == "normal"

    sim.remove_fault(injected["fault_id"])
    after_removal = sim.step(ticks=1)

    assert unhealthy_backend_id not in after_removal["load_balancer_backend_server_ids"]
    assert after_removal["load_balancer_backend_skew_ratio"] <= 0.1
    assert after_removal["load_balancer_error_rate_percent"] == 0.0


def test_load_balancer_misconfiguration_removal_restores_pre_fault_config():
    sim = make_sim()
    sim.reset(seed=7)
    sim.start_workload(
        {
            "request_rate_per_second": 4500,
            "placement_strategy": "spread",
            "noise_enabled": False,
        }
    )
    baseline = sim.step(ticks=1)
    baseline_backend_ids = baseline["load_balancer_backend_server_ids"]
    baseline_backend_weights = baseline["load_balancer_backend_weights"]

    injected = sim.inject_fault(
        FaultRequest(
            fault_type="load_balancer_misconfiguration",
            target="load-balancer",
            severity=0.9,
            duration_seconds=300,
        )
    )
    degraded = sim.step(ticks=1)
    assert degraded["load_balancer_error_rate_percent"] > 0.0

    sim.remove_fault(injected["fault_id"])
    recovered = sim.step(ticks=10)

    assert recovered["load_balancer_backend_server_ids"] == baseline_backend_ids
    assert recovered["load_balancer_backend_weights"] == baseline_backend_weights
    assert recovered["load_balancer_routing_policy"] == "round_robin"
    assert recovered["load_balancer_backend_skew_ratio"] == 0.0
    assert recovered["load_balancer_unhealthy_routing_fraction"] == 0.0
    assert recovered["load_balancer_error_rate_percent"] == 0.0
    assert recovered["workload_application_error_rate_percent"] == 0.0
    assert recovered["workload_queue_length"] == 0
    assert recovered["sla_status"] == "normal"


def test_load_balancer_misconfiguration_expiry_restores_pre_fault_config():
    sim = make_sim()
    sim.reset(seed=7)
    sim.start_workload(
        {
            "request_rate_per_second": 4500,
            "placement_strategy": "spread",
            "noise_enabled": False,
        }
    )
    baseline = sim.step(ticks=1)

    sim.inject_fault(
        FaultRequest(
            fault_type="load_balancer_misconfiguration",
            target="load-balancer",
            severity=0.9,
            duration_seconds=2,
        )
    )
    degraded = sim.step(ticks=1)
    expired = sim.step(ticks=1)

    assert degraded["load_balancer_error_rate_percent"] > 0.0
    assert expired["active_faults"] == []
    assert (
        expired["load_balancer_backend_server_ids"]
        == baseline["load_balancer_backend_server_ids"]
    )
    assert (
        expired["load_balancer_backend_weights"]
        == baseline["load_balancer_backend_weights"]
    )
    assert expired["load_balancer_routing_policy"] == "round_robin"
    assert expired["load_balancer_backend_skew_ratio"] == 0.0
    assert expired["load_balancer_unhealthy_routing_fraction"] == 0.0
    assert expired["load_balancer_error_rate_percent"] == 0.0
    assert expired["workload_application_error_rate_percent"] == 0.0
    assert expired["workload_queue_length"] == 0
    assert expired["sla_status"] == "normal"


def test_application_error_fault_sets_error_telemetry_and_expires():
    sim = make_sim()
    sim.reset(seed=7)
    sim.start_workload(
        {
            "request_rate_per_second": 500,
            "placement_strategy": "spread",
            "noise_enabled": False,
        }
    )

    sim.inject_fault(
        FaultRequest(
            fault_type="application_error",
            target="workload",
            severity=0.25,
            duration_seconds=2,
        )
    )
    active = sim.step(ticks=1)
    expired = sim.step(ticks=1)

    assert active["workload_application_error_rate_percent"] == 25.0
    assert active["workload_dropped_requests_per_second"] == 125.0
    assert active["sla_status"] == "violated"
    assert expired["active_faults"] == []
    assert expired["workload_application_error_rate_percent"] == 0.0
    assert expired["workload_dropped_requests_per_second"] == 0.0


def test_thermal_sensor_miscalibration_creates_reported_temperature_disagreement_and_calibrates():
    sim = make_sim()
    target = "rack-r1-row1-02"
    sim.reset(seed=7)
    sim.start_workload(
        {
            "request_rate_per_second": 300,
            "placement_strategy": "spread",
            "noise_enabled": False,
        }
    )
    baseline = sim.state_summary()

    sim.inject_fault(
        FaultRequest(
            fault_type="thermal_sensor_miscalibration",
            target=target,
            severity=0.9,
            duration_seconds=300,
        )
    )
    degraded = sim.step(ticks=1)
    rack = sim._require_rack(target)
    alert_types = {alert["alert_type"] for alert in sim.alerts()}

    assert (
        degraded["max_temperature_sensor_disagreement_c"]
        > baseline["max_temperature_sensor_disagreement_c"]
    )
    assert degraded["temperature_sensor_unhealthy_count"] == 1
    assert rack.reported_inlet_temperature_c > rack.inlet_temperature_c
    assert "thermal_sensor_health" in alert_types

    sim.apply_control(
        ControlRequest(
            action_type="calibrate_sensor", target=target, calibration_offset_c=-12.6
        )
    )
    recovered = sim.step(ticks=1)

    assert recovered["max_temperature_sensor_disagreement_c"] == 0.0
    assert recovered["temperature_sensor_unhealthy_count"] == 0
    assert recovered["thermal_critical"] == 0


def test_power_budget_violation_uses_budget_status_not_hardware_failure():
    sim = make_sim()
    target = "rack-r1-row1-02"
    sim.reset(seed=7)
    sim.start_workload(
        {
            "request_rate_per_second": 8000,
            "placement_strategy": "rack_hotspot",
            "target_rack_id": target,
            "noise_enabled": False,
        }
    )
    baseline = sim.step(ticks=1)

    sim.inject_fault(
        FaultRequest(
            fault_type="power_budget_violation",
            target=target,
            severity=0.7,
            duration_seconds=300,
        )
    )
    degraded = sim.step(ticks=1)
    rack = sim._require_rack(target)

    assert baseline["power_budget_violating_racks"] == 0
    assert degraded["power_budget_violating_racks"] == 1
    assert degraded["max_power_budget_utilization_ratio"] > 1.0
    assert rack.power_budget_status == "violated"
    assert degraded["failed_servers"] == 0
    assert any(alert["alert_type"] == "rack_power_budget" for alert in sim.alerts())


def test_power_budget_violation_remains_observable_on_small_lightly_loaded_rack():
    config = default_config().merged(
        {
            "simulation": {"auto_advance": False},
            "topology": {
                "rooms": 1,
                "rows_per_room": 2,
                "racks_per_row": 2,
                "servers_per_rack": 3,
            },
        }
    )
    sim = DataCenterSimulator(config)
    target = "rack-r1-row1-01"
    sim.reset(seed=7)
    sim.start_workload(
        {
            "request_rate_per_second": 200,
            "placement_strategy": "spread",
            "noise_enabled": False,
        }
    )
    baseline = sim.step(ticks=1)

    injected = sim.inject_fault(
        FaultRequest(
            fault_type="power_budget_violation",
            target=target,
            severity=0.8,
            duration_seconds=300,
        )
    )
    degraded = sim.step(ticks=1)

    assert baseline["power_budget_violating_racks"] == 0
    assert baseline["sla_status"] == "normal"
    assert degraded["power_budget_violating_racks"] == 1
    assert degraded["max_power_budget_utilization_ratio"] > 1.0
    assert degraded["sla_status"] == "violated"

    sim.remove_fault(injected["fault_id"])
    recovered = sim.step(ticks=1)

    assert recovered["power_budget_violating_racks"] == 0
    assert recovered["sla_status"] == "normal"


def test_intermittent_server_failure_flaps_host_health_without_permanent_failure():
    sim = make_sim()
    target = "server-r1-row1-rack02-03"
    sim.reset(seed=7)
    sim.start_workload(
        {
            "request_rate_per_second": 1800,
            "placement_strategy": "rack_hotspot",
            "target_rack_id": "rack-r1-row1-02",
            "noise_enabled": False,
        }
    )

    sim.inject_fault(
        FaultRequest(
            fault_type="intermittent_server_failure",
            target=target,
            severity=1.0,
            duration_seconds=300,
            period_seconds=4,
            duty_cycle=0.5,
        )
    )
    active = sim.step(ticks=1)
    inactive = sim.step(ticks=2)

    assert active["host_health_flapping_count"] == 1
    assert active["failed_servers"] == 1
    assert inactive["host_health_flapping_count"] == 0
    assert sim._require_server(target).health_status_change_count >= 2
    assert any(
        event["event_type"] == "host_health_status_changed"
        for event in sim.recent_events(20)
    )


def test_thermal_throttling_reduces_capacity_without_marking_server_failed():
    sim = make_sim()
    target = "rack-r1-row1-02"
    sim.reset(seed=7)
    sim.start_workload(
        {
            "request_rate_per_second": 1200,
            "workload_class": "ai_inference",
            "placement_strategy": "rack_hotspot",
            "target_rack_id": target,
            "noise_enabled": False,
        }
    )
    baseline = sim.step(ticks=1)

    sim.inject_fault(
        FaultRequest(
            fault_type="thermal_throttling",
            target=target,
            severity=0.9,
            duration_seconds=300,
        )
    )
    degraded = sim.step(ticks=1)

    assert degraded["thermal_throttled_servers"] > 0
    assert degraded["min_thermal_throttle_factor"] < 1.0
    assert (
        degraded["workload_service_capacity_requests_per_second"]
        < baseline["workload_service_capacity_requests_per_second"]
    )
    assert degraded["failed_servers"] == 0
    assert any(
        alert["alert_type"] in {"rack_capacity_throttled", "host_capacity_throttled"}
        for alert in sim.alerts()
    )


def test_tenant_application_error_fault_targets_only_requested_tenant():
    sim = tenant_sim(
        [
            {
                "tenant_id": "tenant-a",
                "request_rate_per_second": 500,
                "placement_strategy": "spread",
            },
            {
                "tenant_id": "tenant-b",
                "request_rate_per_second": 700,
                "placement_strategy": "spread",
            },
        ]
    )

    sim.inject_fault(
        FaultRequest(
            fault_type="application_error",
            target="tenant-b",
            severity=0.5,
            duration_seconds=30,
        )
    )
    summary = sim.step(ticks=1)
    tenants = summary["tenant_summaries"]
    metrics = render_metrics(sim).decode("utf-8")

    assert tenants["tenant-a"]["application_error_rate_percent"] == 0.0
    assert tenants["tenant-a"]["dropped_requests_per_second"] == 0.0
    assert tenants["tenant-b"]["application_error_rate_percent"] == 50.0
    assert tenants["tenant-b"]["dropped_requests_per_second"] == 350.0
    assert tenants["tenant-a"]["sla_status"] == "normal"
    assert tenants["tenant-a"]["sla_violation_count"] == 0
    assert tenants["tenant-b"]["sla_status"] == "violated"
    assert tenants["tenant-b"]["sla_violation_count"] == 1
    assert summary["workload_dropped_requests_per_second"] == 350.0
    assert summary["sla_status"] == "violated"
    assert (
        'dc_twin_workload_tenant_application_error_rate_percent{tenant_id="tenant-b"'
        in metrics
    )


class _CaptureHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(record)


def test_fault_telemetry_exposes_severity_duration_status_in_summary_metrics_and_logs():
    logger = logging.getLogger("dc-twin-test-fault-telemetry")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handler = _CaptureHandler()
    logger.addHandler(handler)
    sim = make_sim()
    sim.logger = logger

    sim.inject_fault(
        FaultRequest(
            fault_type="cooling_degradation",
            target="cooling-unit-1",
            severity=0.2,
            duration_seconds=2,
        )
    )
    sim.inject_fault(
        FaultRequest(
            fault_type="storage_io_saturation",
            target="storage",
            severity=0.8,
            duration_seconds=4,
        )
    )
    summary = sim.state_summary()
    metrics = render_metrics(sim).decode("utf-8")

    cooling_fault = next(
        fault
        for fault in summary["active_faults"]
        if fault["fault_type"] == "cooling_degradation"
    )
    storage_fault = next(
        fault
        for fault in summary["active_faults"]
        if fault["fault_type"] == "storage_io_saturation"
    )
    assert cooling_fault["severity"] == 0.2
    assert cooling_fault["duration_seconds"] == 2
    assert cooling_fault["started_at_sim_time_seconds"] == 0
    assert cooling_fault["status"] == "active"
    assert cooling_fault["remaining_duration_seconds"] == 2
    assert storage_fault["severity"] == 0.8
    assert storage_fault["duration_seconds"] == 4
    assert storage_fault["status"] == "active"
    assert (
        'dc_twin_fault_active{fault_type="cooling_degradation",target="cooling-unit-1",status="active"} 1.0'
        in metrics
    )
    assert (
        'dc_twin_fault_severity{fault_type="storage_io_saturation",target="storage",status="active"} 0.8'
        in metrics
    )
    assert (
        'dc_twin_fault_duration_seconds{fault_type="storage_io_saturation",target="storage",status="active"} 4.0'
        in metrics
    )

    injected_records = [
        record
        for record in handler.records
        if getattr(record, "event_type", "") == "fault_injected"
    ]
    assert injected_records
    assert injected_records[0].details["severity"] == 0.2
    assert injected_records[0].details["duration_seconds"] == 2
    assert injected_records[0].details["status"] == "active"

    sim.step(ticks=2)
    expired_records = [
        record
        for record in handler.records
        if getattr(record, "event_type", "") == "fault_expired"
    ]
    assert expired_records
    assert expired_records[0].details["status"] == "expired"
    assert expired_records[0].details["remaining_duration_seconds"] == 0


def test_fault_metrics_expose_ranges_for_duplicate_type_target_faults():
    sim = make_sim()
    target = "rack-r1-row1-01"

    sim.inject_fault(
        FaultRequest(
            fault_type="network_partition",
            target=target,
            severity=0.2,
            duration_seconds=10,
        )
    )
    sim.step(ticks=1)
    sim.inject_fault(
        FaultRequest(
            fault_type="network_partition",
            target=target,
            severity=0.8,
            duration_seconds=20,
        )
    )
    summary = sim.state_summary()
    metrics = render_metrics(sim).decode("utf-8")
    labels = f'{{fault_type="network_partition",target="{target}",status="active"}}'

    duplicate_faults = [
        fault
        for fault in summary["active_faults"]
        if fault["fault_type"] == "network_partition" and fault["target"] == target
    ]
    assert len(duplicate_faults) == 2
    assert f"dc_twin_fault_active{labels} 2.0" in metrics
    assert f"dc_twin_fault_min_severity{labels} 0.2" in metrics
    assert f"dc_twin_fault_max_severity{labels} 0.8" in metrics
    assert f"dc_twin_fault_min_duration_seconds{labels} 10.0" in metrics
    assert f"dc_twin_fault_max_duration_seconds{labels} 20.0" in metrics
    assert f"dc_twin_fault_min_remaining_duration_seconds{labels} 9.0" in metrics
    assert f"dc_twin_fault_max_remaining_duration_seconds{labels} 20.0" in metrics
    assert f"dc_twin_fault_started_at_sim_time_seconds{labels} 0.0" in metrics
    assert f"dc_twin_fault_last_started_at_sim_time_seconds{labels} 1.0" in metrics


def test_observation_and_telemetry_expose_alerts_logs_metrics_and_config():
    sim = make_sim()
    sim.inject_fault(
        FaultRequest(
            fault_type="cooling_degradation",
            target="cooling-unit-1",
            severity=0.8,
            duration_seconds=30,
        )
    )

    observation = sim.observation(log_limit=5)
    telemetry = sim.telemetry(log_limit=5)
    metrics = render_metrics(sim).decode("utf-8")

    assert observation["sla_status"] == observation["summary"]["sla_status"]
    assert observation["summary"]["active_faults"][0]["severity"] == 0.8
    assert "cooling_degradation" in observation["configuration"]["supported_faults"]
    assert any(alert["alert_type"] == "active_fault" for alert in observation["alerts"])
    assert any(
        event["event_type"] == "fault_injected"
        for event in observation["recent_events"]
    )
    assert telemetry["metrics"]["faults"]["active_count"] == 1
    assert telemetry["configuration"]["config"]["simulation"]["auto_advance"] is False
    assert (
        telemetry["configuration"]["current"] == observation["configuration"]["current"]
    )
    assert any(event["event_type"] == "fault_injected" for event in telemetry["logs"])
    assert {event["episode_id"] for event in telemetry["logs"]} == {
        telemetry["configuration"]["current"]["episode_id"]
    }
    assert "dc_twin_active_alerts" in metrics
    assert (
        'dc_twin_alert_active{alert_type="active_fault",severity="critical",target="cooling-unit-1"} 1.0'
        in metrics
    )


def test_telemetry_event_log_is_scoped_to_current_episode_after_reset():
    sim = make_sim()
    sim.inject_fault(
        FaultRequest(
            fault_type="server_failure",
            target="server-r1-row1-rack01-01",
            severity=1.0,
            duration_seconds=30,
        )
    )

    sim.reset(seed=99)
    events = sim.recent_events(limit=10)

    assert [event["event_type"] for event in events] == ["sim_reset"]
    assert {event["episode_id"] for event in events} == {events[-1]["episode_id"]}
    assert not any(event["event_type"] == "fault_injected" for event in events)
    assert sim.telemetry(log_limit=0, include_config=False)["logs"] == []
    assert "configuration" not in sim.observation(log_limit=0, include_config=False)


def test_observation_and_telemetry_share_current_runtime_configuration():
    sim = make_sim()
    rack_id = "rack-r1-row1-01"
    sim.start_workload(
        {
            "request_rate_per_second": 500,
            "workload_class": "network_heavy",
            "workload_profile_type": "burst",
            "workload_profile_parameters": {
                "baseline_rate_per_second": 100,
                "burst_rate_per_second": 500,
                "burst_start_time_seconds": 0,
                "burst_duration_seconds": 10,
            },
            "placement_strategy": "rack_hotspot",
            "target_rack_id": rack_id,
            "noise_enabled": False,
        }
    )

    observation = sim.observation(log_limit=5, include_config=True)
    telemetry = sim.telemetry(log_limit=5, include_config=True)
    obs_current = observation["configuration"]["current"]
    tel_current = telemetry["configuration"]["current"]

    assert obs_current == tel_current
    assert tel_current["workload"]["request_rate_per_second"] == 500
    assert tel_current["workload"]["configured_request_rate_per_second"] == 500
    assert tel_current["workload"]["workload_class"] == "network_heavy"
    assert tel_current["workload"]["workload_profile_type"] == "burst"
    assert tel_current["workload"]["placement_strategy"] == "rack_hotspot"
    assert tel_current["workload"]["target_rack_id"] == rack_id
    assert (
        tel_current["workload"]["request_rate_per_second"]
        == telemetry["summary"]["workload_configured_request_rate_per_second"]
    )
    assert (
        telemetry["configuration"]["reset_config"]["workload"][
            "request_rate_per_second"
        ]
        == 100.0
    )
    assert (
        telemetry["configuration"]["config"]
        == telemetry["configuration"]["reset_config"]
    )


def test_telemetry_current_configuration_exposes_tenant_runtime_fields():
    sim = make_sim()
    sim.start_workload(
        {
            "tenants": [
                {
                    "tenant_id": "tenant-a",
                    "request_rate_per_second": 300,
                    "workload_class": "network_heavy",
                    "priority": 2,
                    "quota_requests_per_second": 250,
                    "placement_strategy": "rack_hotspot",
                    "target_rack_id": "rack-r1-row1-01",
                }
            ],
            "noise_enabled": False,
        }
    )

    current = sim.telemetry(log_limit=1, include_config=True)["configuration"][
        "current"
    ]
    tenant_config = current["tenants"]["tenant-a"]

    assert tenant_config["running"] is True
    assert tenant_config["request_rate_per_second"] == 300
    assert tenant_config["workload_class"] == "network_heavy"
    assert tenant_config["priority"] == 2
    assert tenant_config["quota_requests_per_second"] == 250
    assert tenant_config["placement_strategy"] == "rack_hotspot"
    assert tenant_config["target_rack_id"] == "rack-r1-row1-01"


def test_structured_telemetry_exposes_component_local_metrics_matching_prometheus():
    sim = make_sim()
    rack_id = "rack-r1-row1-01"
    server_id = "server-r1-row1-rack01-01"
    cooling_unit_id = "cooling-unit-1"

    sim.inject_fault(
        FaultRequest(
            fault_type="cooling_degradation",
            target=cooling_unit_id,
            severity=0.5,
            duration_seconds=30,
        )
    )
    sim.inject_fault(
        FaultRequest(
            fault_type="rack_hotspot", target=rack_id, severity=0.9, duration_seconds=30
        )
    )
    sim.inject_fault(
        FaultRequest(
            fault_type="server_failure",
            target=server_id,
            severity=1.0,
            duration_seconds=30,
        )
    )

    telemetry = sim.telemetry(log_limit=5, include_config=False)
    metrics = render_metrics(sim).decode("utf-8")
    rack_metrics = telemetry["metrics"]["racks"][rack_id]
    server_metrics = telemetry["metrics"]["servers"][server_id]
    cooling_metrics = telemetry["metrics"]["cooling_units"][cooling_unit_id]

    assert rack_metrics["thermal_status"] in {"warning", "critical"}
    assert server_metrics["status"] == "failed"
    assert server_metrics["failed"] is True
    assert cooling_metrics["status"] == "degraded"
    assert (
        f'dc_twin_rack_inlet_temperature_c{{rack_id="{rack_id}"}} '
        f"{float(rack_metrics['inlet_temperature_c'])}"
    ) in metrics
    assert (
        f'dc_twin_server_failed{{server_id="{server_id}",rack_id="{rack_id}"}} 1.0'
        in metrics
    )
    assert (
        f'dc_twin_cooling_capacity_kw{{cooling_unit_id="{cooling_unit_id}"}} '
        f"{float(cooling_metrics['cooling_capacity_kw'])}"
    ) in metrics


def test_prometheus_workload_running_and_active_class_semantics():
    sim = make_sim()
    idle_metrics = render_metrics(sim).decode("utf-8")

    assert "dc_twin_workload_running 0.0" in idle_metrics
    assert 'dc_twin_workload_profile_active{profile_type="steady"} 0.0' in idle_metrics
    assert (
        'dc_twin_workload_class_active{workload_class="web_service"} 0.0'
        in idle_metrics
    )

    sim.start_workload(
        {
            "request_rate_per_second": 500,
            "workload_class": "network_heavy",
            "noise_enabled": False,
        }
    )
    running_metrics = render_metrics(sim).decode("utf-8")

    assert "dc_twin_workload_running 1.0" in running_metrics
    assert (
        'dc_twin_workload_profile_active{profile_type="steady"} 1.0' in running_metrics
    )
    assert (
        'dc_twin_workload_class_active{workload_class="network_heavy"} 1.0'
        in running_metrics
    )

    sim.stop_workload()
    stopped_metrics = render_metrics(sim).decode("utf-8")

    assert "dc_twin_workload_running 0.0" in stopped_metrics
    assert (
        'dc_twin_workload_class_active{workload_class="network_heavy"} 0.0'
        in stopped_metrics
    )


def workload_summary(payload):
    sim = make_sim()
    sim.reset(seed=7)
    sim.start_workload({"noise_enabled": False, **payload})
    sim.step(ticks=1)
    return sim.state_summary()


def test_low_workload_has_low_latency_and_empty_queue():
    summary = workload_summary(
        {"request_rate_per_second": 100, "placement_strategy": "spread"}
    )

    assert summary["workload_queue_length"] == 0
    assert summary["workload_average_latency_ms"] < 10
    assert summary["workload_p95_latency_ms"] < 15


def test_high_workload_increases_latency_over_low_workload():
    low = workload_summary(
        {"request_rate_per_second": 100, "placement_strategy": "spread"}
    )
    high = workload_summary(
        {"request_rate_per_second": 70000, "placement_strategy": "spread"}
    )

    assert (
        high["average_cpu_utilization_percent"] > low["average_cpu_utilization_percent"]
    )
    assert high["workload_p95_latency_ms"] > low["workload_p95_latency_ms"]


def test_overload_causes_queue_growth_and_latency_growth():
    sim = make_sim()
    sim.reset(seed=7)
    sim.start_workload(
        {
            "request_rate_per_second": 15000,
            "placement_strategy": "rack_hotspot",
            "target_rack_id": "rack-r1-row1-01",
            "noise_enabled": False,
        }
    )
    first = sim.step(ticks=1)
    second = sim.step(ticks=1)

    assert first["workload_queue_length"] > 0
    assert second["workload_queue_length"] > first["workload_queue_length"]
    assert second["workload_p95_latency_ms"] > first["workload_p95_latency_ms"]


def test_queue_decays_when_workload_drops_below_capacity():
    sim = make_sim()
    sim.reset(seed=7)
    sim.start_workload(
        {
            "request_rate_per_second": 15000,
            "placement_strategy": "rack_hotspot",
            "target_rack_id": "rack-r1-row1-01",
            "noise_enabled": False,
        }
    )
    overloaded = sim.step(ticks=3)
    sim.update_workload({"request_rate_per_second": 1000})
    recovered = sim.step(ticks=1)

    assert overloaded["workload_queue_length"] > 0
    assert recovered["workload_queue_length"] < overloaded["workload_queue_length"]
