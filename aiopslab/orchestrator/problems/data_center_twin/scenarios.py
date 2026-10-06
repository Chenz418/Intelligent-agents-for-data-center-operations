"""Scenario manifest loader for Data Center Twin benchmark problems."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


SCENARIOS_PATH = Path(__file__).with_name("scenarios.json")

SUPPORTED_BENCHMARK_FAULTS = {
    "cooling_degradation",
    "network_congestion_burst",
    "network_partition",
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
    "tor_packet_loss",
}
# Evaluator-internal catalog.  It validates the fixed benchmark suite and is
# deliberately not rendered into an agent-facing prompt.
BENCHMARK_FAULT_TAXONOMY = tuple(sorted(SUPPORTED_BENCHMARK_FAULTS))


def fault_diagnosis_contract_lines() -> tuple[str, ...]:
    """Return the taxonomy-free Detection/RCA diagnosis contract."""

    return (
        "Give one concise, free-form diagnosis of the most likely underlying "
        "fault mechanism, not merely an alert name, metric name, operational "
        "domain, or observed symptom.",
        "Describe what is failing or misconfigured and the nature of that "
        "failure in your own words; do not enumerate multiple alternatives.",
        "Use visible alert names, anomalous metrics, logs, traces, and "
        "configuration changes as supporting `evidence`.",
    )


SUPPORTED_TASK_TYPES = {"detection", "localization", "analysis", "mitigation"}
REQUIRED_TASK_TYPES_BY_FAULT = {
    fault_type: frozenset(SUPPORTED_TASK_TYPES)
    for fault_type in SUPPORTED_BENCHMARK_FAULTS
}
SUPPORTED_WORKLOAD_CLASSES = {
    "web_service",
    "ai_training",
    "ai_inference",
    "storage_heavy",
    "network_heavy",
}
SUPPORTED_WORKLOAD_PROFILES = {"steady", "burst", "diurnal", "maintenance_window"}
SUPPORTED_PLACEMENT_STRATEGIES = {"spread", "rack_hotspot", "random"}
READ_TIME_ACTIONS = {"observe", "noop", "step"}
WRITE_ACTIONS = {
    "calibrate_sensor",
    "set_cooling",
    "migrate_workload",
    "throttle_workload",
    "set_server_maintenance",
    "clear_server_maintenance",
    "update_autoscaler_policy",
    "repair_monitoring_pipeline",
    "update_placement_policy",
    "update_load_balancer_config",
}
AGENT_ACTIONS = READ_TIME_ACTIONS | WRITE_ACTIONS


class ScenarioValidationError(ValueError):
    """Raised when the Data Center Twin scenario manifest is invalid."""


@dataclass(frozen=True)
class DataCenterTwinScenario:
    problem_id: str
    class_name: str
    task_type: str
    seed: int
    config_override: dict[str, Any]
    workload: dict[str, Any]
    stabilization_ticks: int
    post_injection_ticks: int
    fault_type: str
    fault_target: str
    fault_severity: float
    fault_duration_seconds: int
    fault_parameters: dict[str, Any]
    allowed_agent_actions: frozenset[str]
    expected: dict[str, list[str]]
    success_criteria: dict[str, Any]


def load_scenarios(path: Path = SCENARIOS_PATH) -> dict[str, DataCenterTwinScenario]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    records = raw.get("scenarios")
    if not isinstance(records, list):
        raise ScenarioValidationError("scenario manifest must contain a scenarios list")

    scenarios: dict[str, DataCenterTwinScenario] = {}
    for index, record in enumerate(records):
        scenario = _parse_scenario(record, index)
        if scenario.problem_id in scenarios:
            raise ScenarioValidationError(
                f"duplicate scenario problem_id: {scenario.problem_id}"
            )
        scenarios[scenario.problem_id] = scenario
    return scenarios


def get_scenario(problem_id: str) -> DataCenterTwinScenario:
    try:
        return SCENARIOS[problem_id]
    except KeyError as error:
        raise ScenarioValidationError(
            f"unknown Data Center Twin scenario: {problem_id}"
        ) from error


def validate_scenario_manifest(
    scenarios: dict[str, DataCenterTwinScenario] | None = None,
    supported_faults: set[str] | frozenset[str] | None = None,
) -> dict[str, DataCenterTwinScenario]:
    scenario_map = SCENARIOS if scenarios is None else scenarios
    expected_faults = set(supported_faults or SUPPORTED_BENCHMARK_FAULTS)
    scenario_faults = {scenario.fault_type for scenario in scenario_map.values()}
    missing_faults = sorted(expected_faults - scenario_faults)
    if missing_faults:
        raise ScenarioValidationError(
            f"scenario manifest does not cover fault types: {missing_faults}"
        )

    unsupported_faults = sorted(scenario_faults - expected_faults)
    if unsupported_faults:
        raise ScenarioValidationError(
            f"scenario manifest contains unsupported fault types: {unsupported_faults}"
        )
    required_task_types = {
        fault_type: set(
            REQUIRED_TASK_TYPES_BY_FAULT.get(fault_type, SUPPORTED_TASK_TYPES)
        )
        for fault_type in expected_faults
    }
    task_types_by_fault: dict[str, set[str]] = {
        fault_type: set() for fault_type in expected_faults
    }
    for scenario in scenario_map.values():
        task_types_by_fault.setdefault(scenario.fault_type, set()).add(
            scenario.task_type
        )
    missing_task_coverage = {
        fault_type: sorted(required_tasks - task_types_by_fault.get(fault_type, set()))
        for fault_type, required_tasks in sorted(required_task_types.items())
        if required_tasks - task_types_by_fault.get(fault_type, set())
    }
    if missing_task_coverage:
        raise ScenarioValidationError(
            f"scenario manifest does not cover required task types by fault: {missing_task_coverage}"
        )
    expected_count = sum(len(tasks) for tasks in required_task_types.values())
    if len(scenario_map) != expected_count:
        raise ScenarioValidationError(
            f"scenario manifest must contain exactly {expected_count} tasks "
            "with one instance per fault/objective pair"
        )
    return scenario_map


def _parse_scenario(record: Any, index: int) -> DataCenterTwinScenario:
    if not isinstance(record, dict):
        raise ScenarioValidationError(f"scenario at index {index} must be an object")

    problem_id = _required_str(record, "problem_id")
    class_name = _required_str(record, "class_name")
    task_type = _required_str(record, "task_type")
    if task_type not in SUPPORTED_TASK_TYPES:
        raise ScenarioValidationError(
            f"{problem_id}: unsupported task_type {task_type}"
        )

    fault = _required_dict(record, "fault")
    fault_type = _required_str(fault, "type")
    fault_target = _required_str(fault, "target")
    fault_severity = _required_number(fault, "severity")
    fault_duration_seconds = _required_int(fault, "duration_seconds")
    fault_parameters = _optional_fault_parameters(problem_id, fault_type, fault)
    if fault_type not in SUPPORTED_BENCHMARK_FAULTS:
        raise ScenarioValidationError(
            f"{problem_id}: unsupported fault type {fault_type}"
        )
    if not 0.0 <= fault_severity <= 1.0:
        raise ScenarioValidationError(
            f"{problem_id}: fault severity must be between 0 and 1"
        )
    if fault_duration_seconds <= 0:
        raise ScenarioValidationError(
            f"{problem_id}: fault duration_seconds must be positive"
        )
    _validate_fault_target(problem_id, fault_type, fault_target)

    seed = _required_int(record, "seed")
    config_override = _required_dict(record, "config_override")
    simulation_config = config_override.get("simulation")
    if (
        not isinstance(simulation_config, dict)
        or simulation_config.get("auto_advance") is not False
    ):
        raise ScenarioValidationError(
            f"{problem_id}: benchmark scenarios must set simulation.auto_advance=false"
        )

    workload = _required_dict(record, "workload")
    _validate_workload(problem_id, workload)

    stabilization_ticks = _required_int(record, "stabilization_ticks")
    if stabilization_ticks < 0:
        raise ScenarioValidationError(
            f"{problem_id}: stabilization_ticks must be non-negative"
        )
    post_injection_ticks = record.get("post_injection_ticks", 0)
    if (
        not isinstance(post_injection_ticks, int)
        or isinstance(post_injection_ticks, bool)
        or post_injection_ticks < 0
    ):
        raise ScenarioValidationError(
            f"{problem_id}: post_injection_ticks must be a non-negative integer"
        )

    allowed_agent_actions = record.get("allowed_agent_actions")
    if not isinstance(allowed_agent_actions, list) or not allowed_agent_actions:
        raise ScenarioValidationError(
            f"{problem_id}: allowed_agent_actions must be a non-empty list"
        )
    allowed_actions = frozenset(
        _coerce_str_list(allowed_agent_actions, f"{problem_id}: allowed_agent_actions")
    )
    unknown_actions = sorted(allowed_actions - AGENT_ACTIONS)
    if unknown_actions:
        raise ScenarioValidationError(
            f"{problem_id}: unknown allowed_agent_actions {unknown_actions}"
        )
    if task_type != "mitigation" and allowed_actions - READ_TIME_ACTIONS:
        raise ScenarioValidationError(
            f"{problem_id}: read-oriented tasks may only expose read/time actions"
        )
    if task_type == "mitigation" and not (allowed_actions & WRITE_ACTIONS):
        raise ScenarioValidationError(
            f"{problem_id}: mitigation tasks must expose at least one write action"
        )

    expected = _required_dict(record, "expected")
    normalized_expected: dict[str, list[str]] = {}
    for key in ("fault_type_terms", "target_terms", "domain_terms", "evidence_terms"):
        values = expected.get(key)
        if not isinstance(values, list) or not values:
            raise ScenarioValidationError(
                f"{problem_id}: expected.{key} must be a non-empty list"
            )
        normalized_expected[key] = _coerce_str_list(
            values, f"{problem_id}: expected.{key}"
        )

    success_criteria = _required_dict(record, "success_criteria")
    if task_type == "mitigation":
        _validate_mitigation_success_criteria(problem_id, success_criteria)
        declared = success_criteria.get("mitigation_actions") or [
            success_criteria.get("mitigation_action")
        ]
        required = set(success_criteria.get("required_control_actions") or [])
        required.update(action["action_type"] for action in declared if action)
        if required - allowed_actions:
            raise ScenarioValidationError(
                f"{problem_id}: mitigation actions unavailable in allowed_agent_actions: "
                f"{sorted(required - allowed_actions)}"
            )

    return DataCenterTwinScenario(
        problem_id=problem_id,
        class_name=class_name,
        task_type=task_type,
        seed=seed,
        config_override=config_override,
        workload=workload,
        stabilization_ticks=stabilization_ticks,
        post_injection_ticks=post_injection_ticks,
        fault_type=fault_type,
        fault_target=fault_target,
        fault_severity=fault_severity,
        fault_duration_seconds=fault_duration_seconds,
        fault_parameters=fault_parameters,
        allowed_agent_actions=allowed_actions,
        expected=normalized_expected,
        success_criteria=success_criteria,
    )


def _validate_workload(problem_id: str, workload: dict[str, Any]) -> None:
    workload_class = workload.get("workload_class", "web_service")
    if workload_class not in SUPPORTED_WORKLOAD_CLASSES:
        raise ScenarioValidationError(
            f"{problem_id}: unsupported workload_class {workload_class}"
        )

    profile_type = workload.get("workload_profile_type", "steady")
    if profile_type not in SUPPORTED_WORKLOAD_PROFILES:
        raise ScenarioValidationError(
            f"{problem_id}: unsupported workload_profile_type {profile_type}"
        )

    placement_strategy = workload.get("placement_strategy", "spread")
    if placement_strategy not in SUPPORTED_PLACEMENT_STRATEGIES:
        raise ScenarioValidationError(
            f"{problem_id}: unsupported placement_strategy {placement_strategy}"
        )

    if placement_strategy == "rack_hotspot" and not workload.get("target_rack_id"):
        raise ScenarioValidationError(
            f"{problem_id}: rack_hotspot placement requires target_rack_id"
        )


def _validate_fault_target(problem_id: str, fault_type: str, target: str) -> None:
    if fault_type == "cooling_degradation" and not target.startswith("cooling-unit-"):
        raise ScenarioValidationError(
            f"{problem_id}: cooling_degradation target must be a cooling unit"
        )
    if fault_type in {
        "rack_hotspot",
        "power_overload",
        "network_partition",
        "power_budget_violation",
        "tor_packet_loss",
    } and not target.startswith("rack-"):
        raise ScenarioValidationError(
            f"{problem_id}: {fault_type} target must be a rack"
        )
    if fault_type == "network_congestion_burst" and not (
        target in {"workload", "application", "global", "datacenter", "*"}
        or target.startswith("tenant-")
        or target.startswith("rack-")
    ):
        raise ScenarioValidationError(
            f"{problem_id}: network_congestion_burst target must be workload, tenant, or rack scoped"
        )
    if fault_type in {
        "server_failure",
        "intermittent_server_failure",
    } and not target.startswith("server-"):
        raise ScenarioValidationError(
            f"{problem_id}: server_failure target must be a server"
        )
    if fault_type in {"thermal_sensor_miscalibration", "thermal_throttling"} and not (
        target.startswith("rack-") or target.startswith("server-")
    ):
        raise ScenarioValidationError(
            f"{problem_id}: {fault_type} target must be a rack or server"
        )
    if fault_type == "storage_io_saturation" and target not in {
        "storage",
        "storage-subsystem",
        "global",
        "datacenter",
        "*",
    }:
        raise ScenarioValidationError(
            f"{problem_id}: storage_io_saturation target must be storage scoped"
        )
    if fault_type == "control_plane_degradation" and target not in {
        "control-plane",
        "scheduler",
        "api-server",
        "global",
        "datacenter",
        "*",
    }:
        raise ScenarioValidationError(
            f"{problem_id}: control_plane_degradation target must be control-plane scoped"
        )
    if fault_type == "autoscaler_misconfiguration" and target not in {
        "autoscaler",
        "workload",
        "control-plane",
        "scheduler",
        "global",
        "datacenter",
        "*",
    }:
        raise ScenarioValidationError(
            f"{problem_id}: autoscaler_misconfiguration target must be autoscaler scoped"
        )
    if fault_type == "monitoring_pipeline_failure" and target not in {
        "monitoring",
        "monitoring-pipeline",
        "telemetry-pipeline",
        "metrics-pipeline",
        "global",
        "datacenter",
        "*",
    }:
        raise ScenarioValidationError(
            f"{problem_id}: monitoring_pipeline_failure target must be monitoring-pipeline scoped"
        )
    if fault_type == "placement_policy_misconfiguration" and not (
        target.startswith("rack-")
        or target
        in {
            "scheduler",
            "placement-policy",
            "workload",
            "control-plane",
            "global",
            "datacenter",
            "*",
        }
    ):
        raise ScenarioValidationError(
            f"{problem_id}: placement_policy_misconfiguration target must be scheduler or rack scoped"
        )
    if fault_type == "load_balancer_misconfiguration" and target not in {
        "application",
        "workload",
        "load-balancer",
        "frontend",
        "service",
        "global",
        "datacenter",
        "*",
    }:
        raise ScenarioValidationError(
            f"{problem_id}: load_balancer_misconfiguration target must be application or load-balancer scoped"
        )
    if fault_type == "application_error" and not target:
        raise ScenarioValidationError(
            f"{problem_id}: application_error target must be non-empty"
        )


def _optional_fault_parameters(
    problem_id: str, fault_type: str, fault: dict[str, Any]
) -> dict[str, Any]:
    parameters: dict[str, Any] = {}
    allowed_keys = {"type", "target", "severity", "duration_seconds"}
    if fault_type == "intermittent_server_failure":
        allowed_keys |= {"period_seconds", "duty_cycle", "failure_window_seconds"}
    unknown_keys = sorted(set(fault) - allowed_keys)
    if unknown_keys:
        raise ScenarioValidationError(
            f"{problem_id}: unsupported fault parameter(s) {unknown_keys}"
        )
    if "period_seconds" in fault:
        period_seconds = _required_int(fault, "period_seconds")
        if period_seconds <= 0:
            raise ScenarioValidationError(
                f"{problem_id}: period_seconds must be positive"
            )
        parameters["period_seconds"] = period_seconds
    if "failure_window_seconds" in fault:
        failure_window_seconds = _required_int(fault, "failure_window_seconds")
        if failure_window_seconds <= 0:
            raise ScenarioValidationError(
                f"{problem_id}: failure_window_seconds must be positive"
            )
        parameters["failure_window_seconds"] = failure_window_seconds
    if "duty_cycle" in fault:
        duty_cycle = _required_number(fault, "duty_cycle")
        if not 0.0 <= duty_cycle <= 1.0:
            raise ScenarioValidationError(
                f"{problem_id}: duty_cycle must be between 0 and 1"
            )
        parameters["duty_cycle"] = duty_cycle
    if "period_seconds" in parameters and "failure_window_seconds" in parameters:
        if parameters["failure_window_seconds"] > parameters["period_seconds"]:
            raise ScenarioValidationError(
                f"{problem_id}: failure_window_seconds cannot exceed period_seconds"
            )
    return parameters


def _validate_mitigation_success_criteria(
    problem_id: str, success_criteria: dict[str, Any]
) -> None:
    action = success_criteria.get("mitigation_action")
    actions = success_criteria.get("mitigation_actions")
    if action is None and actions is None:
        raise ScenarioValidationError(
            f"{problem_id}: mitigation success_criteria must define mitigation_action or mitigation_actions"
        )
    if action is not None:
        _validate_mitigation_action(problem_id, action)
    if actions is not None:
        if not isinstance(actions, list) or not actions:
            raise ScenarioValidationError(
                f"{problem_id}: mitigation_actions must be a non-empty list"
            )
        for index, item in enumerate(actions):
            _validate_mitigation_action(
                f"{problem_id}: mitigation_actions[{index}]", item
            )

    declared_actions = []
    if action is not None:
        declared_actions.append(action["action_type"])
    if actions is not None:
        declared_actions.extend(item["action_type"] for item in actions)

    passive_recovery_allowed = success_criteria.get("passive_recovery_allowed", False)
    if not isinstance(passive_recovery_allowed, bool):
        raise ScenarioValidationError(
            f"{problem_id}: passive_recovery_allowed must be a boolean"
        )
    required_control_actions = success_criteria.get("required_control_actions")
    if not passive_recovery_allowed:
        if (
            not isinstance(required_control_actions, list)
            or not required_control_actions
        ):
            raise ScenarioValidationError(
                f"{problem_id}: mitigation success_criteria must define required_control_actions"
            )
        required_actions = set(
            _coerce_str_list(
                required_control_actions, f"{problem_id}: required_control_actions"
            )
        )
        unknown_required_actions = sorted(required_actions - WRITE_ACTIONS)
        if unknown_required_actions:
            raise ScenarioValidationError(
                f"{problem_id}: required_control_actions contains unknown write actions {unknown_required_actions}"
            )
        missing_declared_actions = sorted(required_actions - set(declared_actions))
        if missing_declared_actions:
            raise ScenarioValidationError(
                f"{problem_id}: required_control_actions must be declared mitigation actions {missing_declared_actions}"
            )


def _validate_mitigation_action(problem_id: str, action: Any) -> None:
    if not isinstance(action, dict):
        raise ScenarioValidationError(
            f"{problem_id}: mitigation action must be an object"
        )
    action_type = _required_str(action, "action_type")
    if action_type not in WRITE_ACTIONS:
        raise ScenarioValidationError(
            f"{problem_id}: mitigation action must be a write action"
        )
    parameters = action.get("parameters", {})
    if not isinstance(parameters, dict):
        raise ScenarioValidationError(
            f"{problem_id}: mitigation action parameters must be an object"
        )
    if "advance_ticks" in action and (
        not isinstance(action["advance_ticks"], int) or action["advance_ticks"] < 0
    ):
        raise ScenarioValidationError(
            f"{problem_id}: mitigation action advance_ticks must be a non-negative integer"
        )


def _required_dict(record: dict[str, Any], key: str) -> dict[str, Any]:
    value = record.get(key)
    if not isinstance(value, dict):
        raise ScenarioValidationError(f"missing or invalid object field: {key}")
    return value


def _required_str(record: dict[str, Any], key: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value:
        raise ScenarioValidationError(f"missing or invalid string field: {key}")
    return value


def _required_int(record: dict[str, Any], key: str) -> int:
    value = record.get(key)
    if not isinstance(value, int):
        raise ScenarioValidationError(f"missing or invalid integer field: {key}")
    return value


def _required_number(record: dict[str, Any], key: str) -> float:
    value = record.get(key)
    if not isinstance(value, int | float):
        raise ScenarioValidationError(f"missing or invalid number field: {key}")
    return float(value)


def _coerce_str_list(values: list[Any], field_name: str) -> list[str]:
    normalized: list[str] = []
    for value in values:
        if not isinstance(value, str) or not value:
            raise ScenarioValidationError(
                f"{field_name} must contain only non-empty strings"
            )
        normalized.append(value)
    return normalized


SCENARIOS = validate_scenario_manifest(load_scenarios())
