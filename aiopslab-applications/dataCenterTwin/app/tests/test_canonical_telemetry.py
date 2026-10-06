from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re

import pytest

from dc_twin.canonical_telemetry import (
    CHANNELS,
    CanonicalTelemetryRecorder,
    assert_inference_safe,
    validate_canonical_snapshot,
)
from dc_twin.controls import ControlRequest
from dc_twin.faults import FaultRequest
from dc_twin.simulator import DataCenterSimulator, SimulationError


FAULT_CASES = [
    ("cooling_degradation", "cooling-unit-1"),
    ("rack_hotspot", "rack-r1-row1-01"),
    ("power_overload", "rack-r1-row1-01"),
    ("server_failure", "server-r1-row1-rack01-01"),
    ("network_partition", "rack-r1-row1-01"),
    ("storage_io_saturation", "storage"),
    ("control_plane_degradation", "scheduler"),
    ("application_error", "application"),
    ("thermal_sensor_miscalibration", "rack-r1-row1-02"),
    ("power_budget_violation", "rack-r1-row1-02"),
    ("intermittent_server_failure", "server-r1-row1-rack02-03"),
    ("thermal_throttling", "rack-r1-row1-02"),
    ("network_congestion_burst", "workload"),
    ("tor_packet_loss", "rack-r1-row1-03"),
    ("autoscaler_misconfiguration", "autoscaler"),
    ("monitoring_pipeline_failure", "monitoring-pipeline"),
    ("placement_policy_misconfiguration", "rack-r1-row1-04"),
    ("load_balancer_misconfiguration", "load-balancer"),
]


def build_incident(fault_type: str, target: str) -> DataCenterSimulator:
    simulator = DataCenterSimulator()
    workload = {"request_rate_per_second": 400, "noise_enabled": False}
    stabilization_ticks = 3
    if fault_type == "power_budget_violation":
        workload = {
            "request_rate_per_second": 8000,
            "workload_class": "web_service",
            "placement_strategy": "rack_hotspot",
            "target_rack_id": "rack-r1-row1-02",
            "noise_enabled": False,
        }
        stabilization_ticks = 5
    elif fault_type == "network_congestion_burst":
        workload = {
            "request_rate_per_second": 1000,
            "workload_class": "network_heavy",
            "workload_profile_type": "burst",
            "workload_profile_parameters": {
                "baseline_rate_per_second": 1000,
                "burst_rate_per_second": 16000,
                "burst_start_time_seconds": 6,
                "burst_duration_seconds": 40,
            },
            "network_capacity_mbps": 300,
            "placement_strategy": "rack_hotspot",
            "target_rack_id": "rack-r1-row1-03",
            "noise_enabled": False,
        }
        stabilization_ticks = 5
    simulator.start_workload(workload)
    simulator.step(stabilization_ticks)
    simulator.inject_fault(
        FaultRequest(
            fault_type=fault_type,
            target=target,
            severity=1.0,
            duration_seconds=30,
        )
    )
    simulator.step(3)
    return simulator


@pytest.mark.parametrize(("fault_type", "target"), FAULT_CASES)
def test_canonical_snapshot_has_all_channels_without_oracle_labels(fault_type, target):
    snapshot = build_incident(fault_type, target).canonical_snapshot()

    validate_canonical_snapshot(snapshot)
    assert set(snapshot["channels"]) == set(CHANNELS)
    assert all(snapshot["channel_counts"][channel] > 0 for channel in CHANNELS)
    assert {item["channel"] for item in snapshot["observations"]} == set(CHANNELS)
    assert all(
        item["metadata"]["available_at_time_seconds"] <= snapshot["query_time_seconds"]
        for item in snapshot["observations"]
    )
    assert all(
        entity["role"] != "affected"
        for item in snapshot["observations"]
        for entity in item["metadata"]["entities"]
    )

    rendered = json.dumps(snapshot, sort_keys=True).lower()
    assert fault_type not in rendered
    assert target not in rendered if target.startswith("server-") else True
    assert_inference_safe(snapshot)


def test_canonical_query_controls_filter_snapshot_without_mutating_store():
    simulator = build_incident("cooling_degradation", "cooling-unit-1")
    complete_before = simulator.canonical_snapshot()

    filtered = simulator.canonical_snapshot(
        lookback_seconds=2,
        channels={"log", "metric", "config"},
        include_config=False,
        log_limit=1,
    )

    validate_canonical_snapshot(filtered)
    assert filtered["channels"] == ["log", "metric"]
    assert filtered["channel_counts"]["log"] <= 1
    assert set(filtered["channel_counts"]) == {"log", "metric"}
    assert {
        observation["channel"] for observation in filtered["observations"]
    } <= {"log", "metric"}
    assert filtered["window"]["start_time_seconds"] == max(
        0.0,
        filtered["query_time_seconds"] - 2.0,
    )

    complete_after = simulator.canonical_snapshot()
    assert complete_after == complete_before


def test_canonical_log_limit_uses_latest_event_end_and_stable_id():
    simulator = build_incident("cooling_degradation", "cooling-unit-1")
    complete = simulator.canonical_snapshot(channels={"log"})
    limited = simulator.canonical_snapshot(channels={"log"}, log_limit=2)

    expected_ids = [
        item["observation_id"]
        for item in sorted(
            complete["observations"],
            key=lambda item: (
                item["metadata"]["event_end_time_seconds"],
                item["observation_id"],
            ),
            reverse=True,
        )[:2]
    ]
    assert {item["observation_id"] for item in limited["observations"]} == set(
        expected_ids
    )
    validate_canonical_snapshot(limited)


def test_canonical_observation_tuple_and_payload_contract():
    snapshot = build_incident("storage_io_saturation", "storage").canonical_snapshot()
    rendered_snapshot = json.dumps(snapshot, sort_keys=True)
    for internal_key in (
        "base_latency_ms",
        "cpu_cost_per_request",
        "network_capacity_mbps",
        "noise_stddev",
        "storage_capacity_iops",
    ):
        assert internal_key not in rendered_snapshot

    for observation in snapshot["observations"]:
        assert set(observation) == {
            "observation_id",
            "channel",
            "window",
            "payload",
            "metadata",
        }
        assert set(observation["window"]) == {
            "start_time_seconds",
            "end_time_seconds",
            "start_inclusive",
            "end_inclusive",
        }
        metadata = observation["metadata"]
        assert {
            "event_start_time_seconds",
            "event_end_time_seconds",
            "ingest_time_seconds",
            "available_at_time_seconds",
            "available_at_sequence",
            "entities",
            "primary_subsystem",
            "primary_subsystem_provenance",
            "correlation_ids",
            "source_references",
            "data_quality",
        } <= set(metadata)
        assert metadata["data_quality"]["availability_mask"]

    metric = next(item for item in snapshot["observations"] if item["channel"] == "metric")
    assert len(metric["payload"]["timestamps_seconds"]) == len(metric["payload"]["values"])
    assert len(metric["payload"]["values"]) == len(metric["payload"]["missingness_mask"])
    assert metric["payload"]["unit"]
    assert metric["payload"]["statistics"]["count"] > 0

    trace = next(item for item in snapshot["observations"] if item["channel"] == "trace")
    assert {
        "operation",
        "source",
        "destination",
        "latency_ms",
        "status_counts",
        "count",
        "retry_count",
        "critical_path",
    } <= set(trace["payload"])

    config = next(item for item in snapshot["observations"] if item["channel"] == "config")
    assert {
        "path",
        "value",
        "value_type",
        "scope",
        "operation",
        "change_time_seconds",
    } <= set(config["payload"])


def test_nested_workload_log_details_never_expose_exact_server_ids():
    exact_server_id = "server-r1-row1-rack01-01"
    simulator = DataCenterSimulator()
    simulator.start_workload(
        {
            "request_rate_per_second": 400,
            "noise_enabled": False,
            "workload_profile_type": "maintenance_window",
            "workload_profile_parameters": {
                "baseline_rate_per_second": 400,
                "maintenance_start_time_seconds": 0,
                "maintenance_end_time_seconds": 3,
                "affected_server_ids": [exact_server_id],
                "affected_server_workload_fraction": 0.75,
            },
        }
    )
    simulator.step()

    snapshot = simulator.canonical_snapshot()
    rendered = json.dumps(snapshot, sort_keys=True)
    assert exact_server_id not in rendered
    assert re.search(r"\bserver-r\d+-row\d+-rack\d+-\d+\b", rendered) is None
    assert "affected_server_ids" not in rendered
    validate_canonical_snapshot(snapshot)
    assert_inference_safe(snapshot)


def test_delayed_record_is_excluded_until_available():
    recorder = CanonicalTelemetryRecorder("episode-delay", tick_seconds=1)
    recorder.record_event(
        {
            "sequence_id": 1,
            "event_type": "workload_updated",
            "sim_time_seconds": 2,
            "message": "Workload configuration changed",
            "details": {"request_rate_per_second": 500},
        },
        available_at_time_seconds=5,
    )

    before = recorder.snapshot(
        query_time_seconds=4,
        lookback_seconds=10,
        channels={"log"},
    )
    after = recorder.snapshot(
        query_time_seconds=5,
        lookback_seconds=10,
        channels={"log"},
    )

    assert before["observations"] == []
    assert len(after["observations"]) == 1
    assert after["observations"][0]["metadata"]["event_start_time_seconds"] == 2
    assert after["observations"][0]["metadata"]["available_at_time_seconds"] == 5


def test_delayed_state_cannot_influence_earlier_available_config_delta():
    simulator = DataCenterSimulator()
    recorder = CanonicalTelemetryRecorder("episode-delayed-state", tick_seconds=1)
    cooling_unit = simulator.cooling_units[0]

    cooling_unit.fan_speed_percent = 77
    recorder.capture(
        simulator,
        reason="delayed-capture",
        available_at_time_seconds=5,
    )
    simulator.sim_time_seconds = 1
    cooling_unit.fan_speed_percent = 60
    recorder.capture(
        simulator,
        reason="current-capture",
        available_at_time_seconds=1,
    )

    before_delayed_arrival = recorder.snapshot(query_time_seconds=1)
    assert not any(
        item["channel"] == "config"
        and 77
        in {
            item["payload"].get("value"),
            item["payload"].get("previous_value"),
        }
        for item in before_delayed_arrival["observations"]
    )
    fan_config = next(
        item
        for item in before_delayed_arrival["observations"]
        if item["channel"] == "config"
        and item["payload"]["path"].endswith("fan_speed_percent")
    )
    assert fan_config["payload"]["value"] == 60
    assert fan_config["payload"]["previous_value"] is None

    after_delayed_arrival = recorder.snapshot(query_time_seconds=5)
    later_fan_events = [
        item
        for item in after_delayed_arrival["observations"]
        if item["channel"] == "config"
        and item["payload"]["path"].endswith("fan_speed_percent")
    ]
    assert any(item["payload"]["value"] == 77 for item in later_fan_events)
    assert any(item["payload"]["previous_value"] == 77 for item in later_fan_events)


def test_snapshot_is_deterministic_and_return_value_is_immutable_copy():
    simulator = build_incident("network_partition", "rack-r1-row1-01")

    first = simulator.canonical_snapshot()
    second = simulator.canonical_snapshot()
    assert first == second

    first["observations"].clear()
    third = simulator.canonical_snapshot()
    assert third == second
    assert third["observations"]


def test_historical_query_excludes_later_effects():
    simulator = DataCenterSimulator()
    simulator.start_workload({"request_rate_per_second": 400, "noise_enabled": False})
    simulator.step(2)
    baseline = simulator.canonical_snapshot(query_time_seconds=2)

    simulator.step(1)
    simulator.inject_fault(
        FaultRequest(
            fault_type="server_failure",
            target="server-r1-row1-rack01-01",
            severity=1.0,
            duration_seconds=30,
        )
    )
    simulator.step(2)
    replayed = simulator.canonical_snapshot(query_time_seconds=2)

    assert replayed == baseline
    assert not any(
        item["channel"] == "alert"
        and item["payload"].get("alert_type") == "HostHealthCheckFailed"
        for item in replayed["observations"]
    )


def test_same_tick_snapshot_replays_with_logical_watermark():
    simulator = DataCenterSimulator()
    simulator.start_workload({"request_rate_per_second": 400, "noise_enabled": False})
    simulator.step(2)
    before = simulator.canonical_snapshot()

    simulator.inject_fault(
        FaultRequest(
            fault_type="server_failure",
            target="server-r1-row1-rack01-01",
            severity=1.0,
            duration_seconds=30,
        )
    )
    after = simulator.canonical_snapshot()
    replayed = simulator.canonical_snapshot(
        query_time_seconds=before["query_time_seconds"],
        query_watermark_sequence=before["query_watermark_sequence"],
    )

    assert after["query_time_seconds"] == before["query_time_seconds"]
    assert after["query_watermark_sequence"] != before["query_watermark_sequence"]
    assert re.fullmatch(
        r"cut-[0-9a-f]{32}",
        after["query_watermark_sequence"],
    )
    assert after["snapshot_id"] != before["snapshot_id"]
    assert replayed == before


def test_public_cut_tokens_and_source_references_hide_private_event_ordinals():
    simulator = DataCenterSimulator(telemetry_opaque_key=b"test-private-key-" * 2)
    simulator.start_workload({"request_rate_per_second": 400, "noise_enabled": False})
    simulator.step(2)
    before = simulator.canonical_snapshot()
    simulator.inject_fault(
        FaultRequest(
            fault_type="server_failure",
            target="server-r1-row1-rack01-01",
            severity=1.0,
            duration_seconds=30,
        )
    )
    simulator.step()
    active = simulator.canonical_snapshot()

    for snapshot in (before, active):
        cut_token = snapshot["query_watermark_sequence"]
        assert re.fullmatch(r"cut-[0-9a-f]{32}", cut_token)
        assert {
            item["metadata"]["available_at_sequence"]
            for item in snapshot["observations"]
        } == {cut_token}

    # This reconstructs the old public, unkeyed source-reference scheme.  A
    # small brute-force range used to recover visible log ordinals and expose
    # the missing hidden fault lifecycle event.
    old_candidates = set()
    for sequence in range(1, 32):
        serialized = json.dumps(
            {
                "episode_id": active["episode_id"],
                "source": f"log:{sequence}",
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
        old_candidates.add("source-" + hashlib.sha256(serialized).hexdigest()[:20])
    public_log_references = {
        reference
        for item in active["observations"]
        if item["channel"] == "log"
        for reference in item["metadata"]["source_references"]
    }
    assert public_log_references
    assert public_log_references.isdisjoint(old_candidates)


def test_query_cannot_move_beyond_current_episode_time():
    simulator = DataCenterSimulator()
    with pytest.raises(SimulationError, match="current episode history"):
        simulator.canonical_snapshot(query_time_seconds=1)


def test_observable_derived_alert_remains_active_until_recovery():
    simulator = build_incident("control_plane_degradation", "scheduler")
    active = simulator.canonical_snapshot()
    capacity_alert = next(
        item
        for item in active["observations"]
        if item["channel"] == "alert"
        and item["payload"]["alert_type"] == "ServiceCapacityDrop"
    )
    assert capacity_alert["payload"]["status"] == "firing"
    assert capacity_alert["payload"]["duration_seconds"] >= 3

    simulator.clear_faults()
    simulator.step()
    recovered = simulator.canonical_snapshot()
    resolved = next(
        item
        for item in recovered["observations"]
        if item["channel"] == "alert"
        and item["payload"]["alert_type"] == "ServiceCapacityDrop"
    )
    assert resolved["payload"]["status"] == "resolved"


def test_direct_operational_signals_cover_cooling_and_control_plane_roots():
    cooling = build_incident(
        "cooling_degradation",
        "cooling-unit-1",
    ).canonical_snapshot()
    cooling_metrics = [
        item
        for item in cooling["observations"]
        if item["channel"] == "metric"
        and item["payload"]["metric_name"].startswith("cooling.")
    ]
    assert cooling_metrics
    assert {
        item["metadata"]["primary_subsystem"] for item in cooling_metrics
    } == {"cooling"}
    cooling_alert = next(
        item
        for item in cooling["observations"]
        if item["channel"] == "alert"
        and item["payload"]["alert_type"] == "CoolingCapacityDrop"
    )
    assert cooling_alert["metadata"]["primary_subsystem"] == "cooling"
    assert cooling_alert["payload"]["status"] == "firing"

    control_plane = build_incident(
        "control_plane_degradation",
        "scheduler",
    ).canonical_snapshot()
    scheduler_metric = next(
        item
        for item in control_plane["observations"]
        if item["channel"] == "metric"
        and item["payload"]["metric_name"]
        == "control_plane.scheduler_api_latency"
    )
    assert scheduler_metric["metadata"]["primary_subsystem"] == "control_plane"
    assert scheduler_metric["payload"]["values"][-1] >= 20.0
    scheduler_alert = next(
        item
        for item in control_plane["observations"]
        if item["channel"] == "alert"
        and item["payload"]["alert_type"] == "SchedulerAPILatencyHigh"
    )
    assert scheduler_alert["metadata"]["primary_subsystem"] == "control_plane"
    assert scheduler_alert["payload"]["status"] == "firing"
    assert_inference_safe(control_plane)


def test_fixed_short_metric_view_uses_only_prior_causal_history_as_reference():
    simulator = DataCenterSimulator()
    simulator.start_workload({"request_rate_per_second": 400, "noise_enabled": False})
    simulator.step(3)
    simulator.inject_fault(
        FaultRequest(
            fault_type="cooling_degradation",
            target="cooling-unit-1",
            severity=1.0,
            duration_seconds=30,
        )
    )
    simulator.step()
    active = simulator.canonical_snapshot(lookback_seconds=1)
    capacity = next(
        item
        for item in active["observations"]
        if item["channel"] == "metric"
        and item["payload"]["metric_name"] == "cooling.capacity"
        and item["payload"]["resource"]["entity_id"] == "cooling-unit-1"
    )
    reference = capacity["payload"]["normalization_reference"]
    assert reference["method"] == "causal_pre_window_robust"
    assert reference["reference_end_time_seconds"] < capacity["window"][
        "start_time_seconds"
    ]
    assert reference["last"] > capacity["payload"]["values"][-1]

    simulator.clear_faults()
    simulator.step()
    recovered = simulator.canonical_snapshot(lookback_seconds=1)
    recovered_capacity = next(
        item
        for item in recovered["observations"]
        if item["channel"] == "metric"
        and item["payload"]["metric_name"] == "cooling.capacity"
        and item["payload"]["resource"]["entity_id"] == "cooling-unit-1"
    )
    assert (
        recovered_capacity["payload"]["normalization_reference"]["last"]
        < recovered_capacity["payload"]["values"][-1]
    )


def test_monitoring_pipeline_failure_delays_and_drops_canonical_operational_metrics():
    simulator = DataCenterSimulator()
    simulator.reset(seed=7)
    simulator.start_workload(
        {
            "request_rate_per_second": 300,
            "placement_strategy": "spread",
            "noise_enabled": False,
        }
    )
    simulator.step(5)
    simulator.inject_fault(
        FaultRequest(
            fault_type="monitoring_pipeline_failure",
            target="monitoring-pipeline",
            severity=0.85,
            duration_seconds=300,
        )
    )
    simulator.step(3)

    degraded = simulator.canonical_snapshot()
    metrics = {
        item["payload"]["metric_name"]: item
        for item in degraded["observations"]
        if item["channel"] == "metric"
    }
    lag = metrics["observability.telemetry_lag"]["payload"]["values"][-1]
    cutoff = metrics["observability.metrics_last_updated"]["payload"]["values"][-1]
    missing_ratio = metrics["observability.metrics_missing_ratio"]["payload"]["values"][-1]
    latency = metrics["workload.average_latency"]

    assert lag > 0
    assert missing_ratio > 0
    assert cutoff < degraded["query_time_seconds"]
    assert latency["payload"]["values"][-1] is None
    assert any(latency["payload"]["missingness_mask"])
    assert latency["metadata"]["data_quality"]["missingness_fraction"] > 0
    assert all(
        timestamp <= cutoff
        for timestamp, value in zip(
            latency["payload"]["timestamps_seconds"],
            latency["payload"]["values"],
            strict=True,
        )
        if value is not None
    )

    simulator.apply_control(
        ControlRequest(
            action_type="repair_monitoring_pipeline",
            target="monitoring-pipeline",
        )
    )
    simulator.step(1)
    recovered = simulator.canonical_snapshot()
    recovered_metrics = {
        item["payload"]["metric_name"]: item
        for item in recovered["observations"]
        if item["channel"] == "metric"
    }
    recovered_latency = recovered_metrics["workload.average_latency"]

    assert recovered_metrics["observability.telemetry_lag"]["payload"]["values"][-1] == 0
    assert (
        recovered_metrics["observability.metrics_missing_ratio"]["payload"]["values"][-1]
        == 0
    )
    assert recovered_latency["payload"]["timestamps_seconds"][-1] == recovered[
        "query_time_seconds"
    ]
    assert recovered_latency["payload"]["values"][-1] is not None
    recovered_values_by_time = dict(
        zip(
            recovered_latency["payload"]["timestamps_seconds"],
            recovered_latency["payload"]["values"],
            strict=True,
        )
    )
    assert recovered_values_by_time[5.0] is not None
    assert all(recovered_values_by_time[timestamp] is None for timestamp in (6.0, 7.0))
    assert recovered_values_by_time[8.0] is not None
    assert recovered_latency["metadata"]["data_quality"]["missingness_fraction"] > 0


def test_registered_scenario_configs_all_produce_safe_canonical_snapshots():
    repository_root = Path(__file__).resolve().parents[4]
    manifest = json.loads(
        (
            repository_root
            / "aiopslab"
            / "orchestrator"
            / "problems"
            / "data_center_twin"
            / "scenarios.json"
        ).read_text(encoding="utf-8")
    )
    scenarios_by_mechanism = {}
    for scenario in manifest["scenarios"]:
        scenarios_by_mechanism.setdefault(scenario["fault"]["type"], scenario)

    assert set(scenarios_by_mechanism) == {fault_type for fault_type, _ in FAULT_CASES}
    for scenario in scenarios_by_mechanism.values():
        simulator = DataCenterSimulator()
        simulator.reset(
            seed=scenario["seed"],
            config_override=scenario.get("config_override"),
        )
        simulator.start_workload(scenario.get("workload"))
        simulator.step(scenario.get("stabilization_ticks", 0))
        fault = dict(scenario["fault"])
        fault["fault_type"] = fault.pop("type")
        simulator.inject_fault(FaultRequest(**fault))

        snapshot = simulator.canonical_snapshot()
        validate_canonical_snapshot(snapshot)
        assert_inference_safe(snapshot)


def test_task_variants_have_identical_public_prefault_conditions_within_each_mechanism():
    repository_root = Path(__file__).resolve().parents[4]
    manifest = json.loads(
        (
            repository_root
            / "aiopslab"
            / "orchestrator"
            / "problems"
            / "data_center_twin"
            / "scenarios.json"
        ).read_text(encoding="utf-8")
    )
    scenarios_by_mechanism = {}
    for scenario in manifest["scenarios"]:
        scenarios_by_mechanism.setdefault(scenario["fault"]["type"], []).append(
            scenario
        )

    assert len(scenarios_by_mechanism) == len(FAULT_CASES)
    for scenarios in scenarios_by_mechanism.values():
        public_prefault_snapshots = []
        assert {scenario["task_type"] for scenario in scenarios} == {
            "detection",
            "localization",
            "analysis",
            "mitigation",
        }
        for scenario in scenarios:
            simulator = DataCenterSimulator(
                telemetry_opaque_key=b"prefault-comparison-private-key",
            )
            simulator.reset(
                seed=scenario["seed"],
                config_override=scenario.get("config_override"),
            )
            simulator.start_workload(scenario["workload"])
            simulator.step(scenario["stabilization_ticks"])
            public_prefault_snapshots.append(simulator.canonical_snapshot())

        assert all(
            snapshot == public_prefault_snapshots[0]
            for snapshot in public_prefault_snapshots[1:]
        )


def test_added_families_share_counterfactual_prefault_workload_regimes():
    repository_root = Path(__file__).resolve().parents[4]
    manifest = json.loads(
        (
            repository_root
            / "aiopslab"
            / "orchestrator"
            / "problems"
            / "data_center_twin"
            / "scenarios.json"
        ).read_text(encoding="utf-8")
    )
    detection_workloads = {
        scenario["fault"]["type"]: scenario["workload"]
        for scenario in manifest["scenarios"]
        if scenario["task_type"] == "detection"
    }

    # These matched operating regimes ensure the setup alone cannot identify
    # which member of each counterfactual group will be injected. The two
    # network cases remain distinct because one requires an active burst
    # profile while the other requires a steady rack-local path.
    assert (
        detection_workloads["thermal_sensor_miscalibration"]
        == detection_workloads["monitoring_pipeline_failure"]
    )
    assert (
        detection_workloads["power_budget_violation"]
        == detection_workloads["intermittent_server_failure"]
        == detection_workloads["thermal_throttling"]
    )
    assert (
        detection_workloads["autoscaler_misconfiguration"]
        == detection_workloads["placement_policy_misconfiguration"]
        == detection_workloads["load_balancer_misconfiguration"]
    )
    assert (
        len(
            {
                json.dumps(
                    detection_workloads[fault_type],
                    sort_keys=True,
                )
                for fault_type, _target in FAULT_CASES[8:]
            }
        )
        == 5
    )


def test_canonical_validator_rejects_tampered_schema_and_digest():
    snapshot = build_incident("application_error", "application").canonical_snapshot()
    tampered = deepcopy(snapshot)
    tampered["channels"] = ["metric"]
    tampered["channel_counts"] = {"metric": 999}
    tampered["observations"][0]["payload"] = {}
    tampered["observations"][0]["metadata"]["data_quality"]["validation_flags"] = [123]

    with pytest.raises(ValueError):
        validate_canonical_snapshot(tampered)


def _rehash_canonical_snapshot(snapshot):
    without_id = deepcopy(snapshot)
    without_id.pop("snapshot_id", None)
    serialized = json.dumps(
        without_id,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    ).encode("utf-8")
    snapshot["snapshot_id"] = (
        "snapshot-" + hashlib.sha256(serialized).hexdigest()[:24]
    )


def test_both_inference_boundaries_fail_closed_on_schema_mutations():
    snapshot = build_incident(
        "application_error",
        "application",
    ).canonical_snapshot()
    metric_index = next(
        index
        for index, observation in enumerate(snapshot["observations"])
        if observation["channel"] == "metric"
    )
    alert_index = next(
        index
        for index, observation in enumerate(snapshot["observations"])
        if observation["channel"] == "alert"
    )

    def add_top_level_oracle(value):
        value["oracle"] = {"root_cause": "application"}

    def change_policy(value):
        value["policy"]["host_visibility"] = "host"

    def add_observation_field(value):
        value["observations"][0]["training_label"] = "positive"

    def add_payload_field(value):
        value["observations"][metric_index]["payload"]["hidden_score"] = 1.0

    def add_nested_oracle_field(value):
        value["observations"][alert_index]["payload"]["details"][
            "oracle"
        ] = "hidden"

    def change_metric_statistic_type(value):
        value["observations"][metric_index]["payload"]["statistics"]["count"] = "1"

    def add_metadata_field(value):
        value["observations"][0]["metadata"]["future_available"] = True

    def violate_ingest_consistency(value):
        value["observations"][0]["metadata"]["ingest_time_seconds"] += 1.0

    def expose_sequence_counter(value):
        value["query_watermark_sequence"] = 12
        for observation in value["observations"]:
            observation["metadata"]["available_at_sequence"] = 12

    def corrupt_quality_type(value):
        value["observations"][0]["metadata"]["data_quality"][
            "validation_flags"
        ] = [1]

    def remove_source_references(value):
        value["observations"][0]["metadata"]["source_references"] = []

    def remove_entity_provenance(value):
        value["observations"][0]["metadata"]["entities"][0]["provenance"] = ""

    def duplicate_observation_id(value):
        value["observations"][1]["observation_id"] = value["observations"][0][
            "observation_id"
        ]

    mutations = {
        "top-level oracle": add_top_level_oracle,
        "host visibility": change_policy,
        "observation sidecar": add_observation_field,
        "payload extension": add_payload_field,
        "nested oracle field": add_nested_oracle_field,
        "metric statistic type": change_metric_statistic_type,
        "metadata extension": add_metadata_field,
        "ingest inconsistency": violate_ingest_consistency,
        "sequence counter": expose_sequence_counter,
        "quality type": corrupt_quality_type,
        "missing source references": remove_source_references,
        "missing entity provenance": remove_entity_provenance,
        "duplicate observation id": duplicate_observation_id,
    }

    for _name, mutate in mutations.items():
        tampered = deepcopy(snapshot)
        mutate(tampered)
        _rehash_canonical_snapshot(tampered)
        with pytest.raises(ValueError):
            validate_canonical_snapshot(tampered)
