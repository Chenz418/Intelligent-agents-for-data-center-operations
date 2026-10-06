"""Physical thermal degradation must reach workload service capacity."""

import json
from pathlib import Path

import pytest

from dc_twin.config import default_config
from dc_twin.controls import ControlRequest
from dc_twin.faults import FaultRequest
from dc_twin.simulator import DataCenterSimulator


CAPACITY = "workload_service_capacity_requests_per_second"
QUEUE = "workload_queue_length"
LATENCY = "workload_average_latency_ms"
REPO_ROOT = Path(__file__).resolve().parents[4]
COOLING_SCENARIOS = [
    scenario
    for scenario in json.loads(
        (REPO_ROOT / "aiopslab/orchestrator/problems/data_center_twin/scenarios.json").read_text()
    )["scenarios"]
    if scenario["fault"]["type"] == "cooling_degradation"
]


def constrained_sim(*, tenants=False, racks=1, demand=3000):
    sim = DataCenterSimulator(
        default_config().merged(
            {
                "simulation": {"auto_advance": False},
                "topology": {
                    "rooms": 1,
                    "rows_per_room": 1,
                    "racks_per_row": racks,
                    "servers_per_rack": 4,
                },
            }
        )
    )
    sim.reset(seed=7)
    workload = {"noise_enabled": False, "request_rate_per_second": demand}
    if tenants:
        workload["tenants"] = [
            {"tenant_id": "tenant-a", "request_rate_per_second": demand / 2},
            {"tenant_id": "tenant-b", "request_rate_per_second": demand / 2},
        ]
    sim.start_workload(workload)
    sim.step(ticks=1)
    return sim


def inject_cooling(sim, *, severity=1.0, duration_seconds=300):
    return sim.inject_fault(
        FaultRequest(
            fault_type="cooling_degradation",
            target="cooling-unit-1",
            severity=severity,
            duration_seconds=duration_seconds,
        )
    )


@pytest.mark.parametrize("scenario", COOLING_SCENARIOS, ids=lambda scenario: scenario["task_type"])
def test_benchmark_cooling_fault_reduces_capacity_in_current_state(scenario):
    sim = DataCenterSimulator(default_config())
    sim.reset(seed=scenario["seed"], config_override=scenario["config_override"])
    sim.start_workload(scenario["workload"])
    baseline = sim.step(ticks=scenario["stabilization_ticks"])
    fault = scenario["fault"]
    sim.inject_fault(
        FaultRequest(
            fault_type=fault["type"],
            target=fault["target"],
            severity=fault["severity"],
            duration_seconds=fault["duration_seconds"],
        )
    )
    degraded = sim.state_summary()

    assert degraded["sim_time_seconds"] == baseline["sim_time_seconds"]
    assert degraded["max_rack_inlet_temperature_c"] > baseline["max_rack_inlet_temperature_c"]
    assert degraded["thermal_throttled_servers"] > 0
    assert degraded[CAPACITY] < baseline[CAPACITY]
    assert degraded["failed_servers"] == 0
    assert degraded[QUEUE] == baseline[QUEUE]


def test_cooling_capacity_loss_grows_queue_and_latency_under_constrained_demand():
    sim = constrained_sim()
    baseline = sim.state_summary()
    assert baseline[CAPACITY] > baseline["workload_current_demand_per_second"]
    assert baseline[QUEUE] == 0

    inject_cooling(sim)
    degraded = sim.step(ticks=1)
    assert degraded[CAPACITY] < degraded["workload_current_demand_per_second"]
    assert degraded[QUEUE] > baseline[QUEUE]
    assert degraded[LATENCY] > baseline[LATENCY]
    assert degraded["thermal_critical"] > 0
    assert all(server.cpu_frequency_scale == server.thermal_throttle_factor for server in sim.servers)


@pytest.mark.parametrize("recovery", ["removal", "expiry", "cooling_control"])
def test_cooling_capacity_recovers_after_fault_or_temperature_recovers(recovery):
    sim = constrained_sim(demand=1000)
    baseline = sim.state_summary()
    injected = inject_cooling(sim, duration_seconds=2 if recovery == "expiry" else 300)
    degraded = sim.step(ticks=1)
    assert degraded[CAPACITY] < baseline[CAPACITY]

    if recovery == "removal":
        sim.remove_fault(injected["fault_id"])
    elif recovery == "expiry":
        sim.step(ticks=1)
    else:
        sim.apply_control(
            ControlRequest(
                action_type="set_cooling",
                target="cooling-unit-1",
                fan_speed_percent=100,
                supply_air_temperature_c=16,
            )
        )
    recovered = sim.state_summary()
    assert recovered[CAPACITY] > degraded[CAPACITY]
    assert recovered["max_rack_inlet_temperature_c"] < degraded["max_rack_inlet_temperature_c"]
    if recovery != "cooling_control":
        assert recovered[CAPACITY] == pytest.approx(baseline[CAPACITY])
        assert recovered["thermal_throttled_servers"] == 0


def test_shared_tenant_capacity_respects_thermally_reduced_server_pool():
    sim = constrained_sim(tenants=True)
    baseline = sim.state_summary()
    inject_cooling(sim)
    degraded = sim.step(ticks=1)

    for tenant_id, tenant in degraded["tenant_summaries"].items():
        assert tenant["service_capacity_requests_per_second"] < baseline["tenant_summaries"][tenant_id][
            "service_capacity_requests_per_second"
        ]
    # Both tenants compete for the same four physical servers. Cooling must not
    # give each tenant a separate copy of the reduced CPU budget.
    processed = sum(tenant["processed_rate_per_second"] for tenant in degraded["tenant_summaries"].values())
    physical_capacity = sum(server.thermal_throttle_factor for server in sim.servers) / sim.workload.cpu_cost_per_request
    assert processed == pytest.approx(physical_capacity, abs=0.001)
    assert degraded[QUEUE] > baseline[QUEUE]
    assert degraded[LATENCY] > baseline[LATENCY]


def test_mild_cooling_degradation_below_warning_does_not_reduce_capacity():
    sim = constrained_sim()
    baseline = sim.state_summary()
    inject_cooling(sim, severity=0.1)
    observed = sim.state_summary()

    assert baseline["max_rack_inlet_temperature_c"] < observed["max_rack_inlet_temperature_c"]
    assert observed["max_rack_inlet_temperature_c"] < sim.config.thresholds.rack_warning_temp_c
    assert observed["thermal_throttled_servers"] == 0
    assert observed[CAPACITY] == baseline[CAPACITY]


def test_rack_hotspot_only_derates_physically_hot_rack():
    sim = constrained_sim(racks=2)
    baseline = sim.state_summary()
    target = sim.racks[0].rack_id
    sim.inject_fault(FaultRequest(fault_type="rack_hotspot", target=target, severity=0.8, duration_seconds=300))

    assert all(server.thermal_throttle_factor < 1 for server in sim.servers if server.rack_id == target)
    assert all(server.thermal_throttle_factor == 1 for server in sim.servers if server.rack_id != target)
    assert sim.state_summary()[CAPACITY] < baseline[CAPACITY]


@pytest.mark.parametrize("target_kind", ["rack", "server"])
def test_sensor_bias_does_not_reduce_physical_compute_capacity(target_kind):
    sim = constrained_sim()
    baseline = sim.state_summary()
    target = sim.racks[0].rack_id if target_kind == "rack" else sim.servers[0].server_id
    sim.inject_fault(
        FaultRequest(fault_type="thermal_sensor_miscalibration", target=target, severity=1.0, duration_seconds=300)
    )
    observed = sim.state_summary()

    assert observed["max_temperature_sensor_disagreement_c"] > 0
    if target_kind == "rack":
        assert observed["thermal_critical"] > 0
    assert observed["max_rack_inlet_temperature_c"] == baseline["max_rack_inlet_temperature_c"]
    assert observed[CAPACITY] == baseline[CAPACITY]
    assert observed["thermal_throttled_servers"] == 0


def test_explicit_thermal_throttling_composes_without_double_derating():
    sim = constrained_sim(demand=1000)
    sim.inject_fault(
        FaultRequest(fault_type="thermal_throttling", target=sim.racks[0].rack_id, severity=0.5, duration_seconds=300)
    )
    explicitly_throttled = sim.state_summary()
    inject_cooling(sim, severity=0.1)
    mildly_hotter = sim.state_summary()
    assert mildly_hotter["max_rack_inlet_temperature_c"] > explicitly_throttled["max_rack_inlet_temperature_c"]
    assert mildly_hotter[CAPACITY] == pytest.approx(explicitly_throttled[CAPACITY])

    inject_cooling(sim, severity=1.0)
    severely_hotter = sim.state_summary()
    assert severely_hotter[CAPACITY] < explicitly_throttled[CAPACITY]
    assert severely_hotter["min_thermal_throttle_factor"] >= 0.2


@pytest.mark.parametrize("tenants", [False, True], ids=["single_workload", "shared_tenants"])
def test_thermal_recalculation_integrates_queue_once_per_tick(tenants):
    def run():
        sim = constrained_sim(tenants=tenants)
        inject_cooling(sim)
        previous_queue = sim.state_summary()[QUEUE]
        snapshots = []
        for _ in range(2):
            summary = sim.step(ticks=1)
            if tenants:
                processed = sum(t["processed_rate_per_second"] for t in summary["tenant_summaries"].values())
            else:
                processed = summary[CAPACITY]
            expected_queue = previous_queue + (
                summary["workload_current_demand_per_second"] - processed
            ) * sim.config.simulation.tick_seconds
            assert summary[QUEUE] == pytest.approx(round(expected_queue), abs=1)
            previous_queue = summary[QUEUE]

            # Applying the current cooling settings forces recalculation at the
            # same simulated time; neither a solver pass nor a control is a tick.
            sim.apply_control(
                ControlRequest(
                    action_type="set_cooling",
                    target="cooling-unit-1",
                    fan_speed_percent=60,
                    supply_air_temperature_c=20,
                )
            )
            unchanged = sim.state_summary()
            assert unchanged[QUEUE] == summary[QUEUE]
            assert unchanged["sim_time_seconds"] == summary["sim_time_seconds"]
            snapshots.append((summary[CAPACITY], summary[QUEUE], summary[LATENCY]))
        return snapshots

    assert run() == run()


def test_thermal_solver_preserves_random_placement_noise_and_trace_progress(tmp_path):
    trace_path = tmp_path / "changing_workload.jsonl"
    trace_path.write_text(
        "\n".join(
            json.dumps(record)
            for record in [
                {"timestamp": 0, "workload_class": "web_service", "request_rate": 1500},
                {"timestamp": 1, "workload_class": "ai_inference", "request_rate": 1000},
                {"timestamp": 2, "workload_class": "web_service", "request_rate": 2500},
            ]
        )
    )
    config = default_config().merged(
        {
            "simulation": {"auto_advance": False},
            "topology": {"rooms": 1, "rows_per_room": 1, "racks_per_row": 2, "servers_per_rack": 4},
        }
    )
    healthy, hot = DataCenterSimulator(config), DataCenterSimulator(config)
    for sim, severity in [(healthy, 0.0), (hot, 1.0)]:
        sim.reset(seed=19)
        inject_cooling(sim, severity=severity)
        sim.start_workload(
            {
                "placement_strategy": "random",
                "noise_enabled": True,
                "trace_replay": {"path": str(trace_path)},
            }
        )

    for tick in range(3):
        expected, observed = healthy.state_summary(), hot.state_summary()
        assert observed["workload_desired_allocation_weights"] == expected["workload_desired_allocation_weights"]
        assert observed["workload_current_demand_per_second"] == expected["workload_current_demand_per_second"]
        assert hot.workload.trace_replay_progress == healthy.workload.trace_replay_progress
        assert hot.workload.trace_replay_progress["current_index"] == tick
        assert hot.rng.getstate() == healthy.rng.getstate()
        assert observed[CAPACITY] < expected[CAPACITY]
        if tick < 2:
            healthy.step(ticks=1)
            hot.step(ticks=1)


def test_thermal_derating_excludes_failed_and_maintenance_servers():
    sim = constrained_sim(demand=1000)
    failed, maintenance = sim.servers[:2]
    sim.inject_fault(
        FaultRequest(fault_type="server_failure", target=failed.server_id, severity=1.0, duration_seconds=300)
    )
    sim.apply_control(ControlRequest(action_type="set_server_maintenance", server_id=maintenance.server_id))
    baseline = sim.state_summary()
    inject_cooling(sim)
    degraded = sim.step(ticks=1)

    assert failed.status == "failed"
    assert maintenance.status == "maintenance"
    assert failed.power_kw == 0
    assert maintenance.power_kw == sim.config.server.idle_power_kw * 0.5
    for server in [failed, maintenance]:
        assert server.thermal_throttle_factor == 1
        assert server.cpu_frequency_scale == 1
        assert server.cpu_utilization_percent == 0
        assert server.workload_assigned == 0
        assert server.server_id not in degraded["workload_allocated_server_ids"]
    assert degraded["thermal_throttled_servers"] == 2
    assert degraded[CAPACITY] < baseline[CAPACITY]
    assert degraded[QUEUE] > baseline[QUEUE]
    assert sim.total_it_power_kw == pytest.approx(sum(server.power_kw for server in sim.servers), abs=0.0001)


def test_power_overload_reduces_capacity_when_physical_heat_crosses_warning():
    sim = constrained_sim()
    sim.reset(
        seed=7,
        config_override={"thresholds": {"rack_warning_temp_c": 23, "rack_critical_temp_c": 28}},
    )
    sim.start_workload({"request_rate_per_second": 3800, "noise_enabled": False})
    baseline = sim.step(ticks=1)
    assert baseline["thermal_throttled_servers"] == 0
    assert baseline[QUEUE] == 0
    sim.inject_fault(
        FaultRequest(fault_type="power_overload", target=sim.racks[0].rack_id, severity=1.0, duration_seconds=300)
    )
    degraded = sim.step(ticks=1)

    assert degraded["max_rack_inlet_temperature_c"] > sim.config.thresholds.rack_warning_temp_c
    assert degraded["power_overloaded_racks"] > 0
    assert degraded["thermal_throttled_servers"] > 0
    assert degraded[CAPACITY] < baseline[CAPACITY]
    assert degraded[QUEUE] > 0
