import hashlib
import json
import re

import pytest
from pydantic import ValidationError

from aiopslab.orchestrator.problems.data_center_twin.visibility import (
    assert_no_agent_leakage,
)
from dc_twin.agent_interface import agent_metric_history
from dc_twin.config import default_config
from dc_twin.faults import FaultRequest
from dc_twin.metric_history import MetricPoint, MetricSeries, metric_catalog_records
from dc_twin.simulator import DataCenterSimulator


def make_history_sim() -> DataCenterSimulator:
    sim = DataCenterSimulator(default_config())
    sim.reset(
        seed=123,
        config_override={
            "simulation": {"auto_advance": False},
            "topology": {
                "rooms": 1,
                "rows_per_room": 1,
                "racks_per_row": 1,
                "servers_per_rack": 2,
            },
            "cooling": {"units": 1},
        },
    )
    return sim


def dump_series(series):
    return [item.model_dump(mode="json") for item in series]


def history_hash(series) -> str:
    encoded = json.dumps(
        dump_series(series), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def all_points(series):
    return [point for item in dump_series(series) for point in item["points"]]


def all_records(series):
    return [
        {
            "entity_id": item["entity_id"],
            "entity_type": item["entity_type"],
            "metric_name": item["metric_name"],
            "unit": item["unit"],
            "metric_kind": item["metric_kind"],
            **point,
        }
        for item in dump_series(series)
        for point in item["points"]
    ]


def test_metric_models_are_strict():
    point = MetricPoint(
        sim_time_seconds=0,
        value=1.0,
    )
    series = MetricSeries(
        metric_name="facility_power_kw",
        entity_id="datacenter",
        entity_type="datacenter",
        unit="kilowatt",
        metric_kind="gauge",
        points=[point],
    )

    assert point.model_dump() == {"sim_time_seconds": 0, "value": 1.0}
    assert series.metric_name == "facility_power_kw"
    with pytest.raises(ValidationError):
        MetricPoint(
            sim_time_seconds=0,
            value=1.0,
            unexpected=True,
        )


def test_reset_plus_three_steps_produces_four_timestamps_and_no_duplicates():
    sim = make_history_sim()
    sim.step(3)

    series = sim.query_metric_history(0, 4)
    points = all_records(series)

    assert sorted({point["sim_time_seconds"] for point in points}) == [0, 1, 2, 3]
    keys = [
        (point["sim_time_seconds"], point["entity_id"], point["metric_name"])
        for point in points
    ]
    assert len(keys) == len(set(keys))


def test_metric_history_reads_do_not_mutate_history():
    sim = make_history_sim()
    sim.step(3)
    before = history_hash(sim.query_metric_history(0, 4))

    sim.state_summary()
    sim.observation(log_limit=10, include_config=True)
    sim.telemetry(log_limit=10, include_config=True)
    sim.query_metric_history(0, 4)
    sim.query_metric_history(0, 4, entity_id="datacenter")

    after = history_hash(sim.query_metric_history(0, 4))
    assert after == before


def test_metric_history_uses_half_open_boundaries():
    sim = make_history_sim()
    sim.step(3)

    series = sim.query_metric_history(
        1, 3, entity_id="datacenter", metric_name="facility_power_kw"
    )
    assert len(series) == 1
    assert [point.sim_time_seconds for point in series[0].points] == [1, 2]

    empty = sim.query_metric_history(
        3, 3, entity_id="datacenter", metric_name="facility_power_kw"
    )
    assert empty == []


def test_metric_history_entity_type_entity_id_and_metric_filters_work():
    sim = make_history_sim()
    sim.step(2)

    by_metric = sim.query_metric_history(0, 3, metric_name="facility_power_kw")
    by_entity = sim.query_metric_history(0, 3, entity_id="datacenter")
    by_type = sim.query_metric_history(0, 3, entity_type="rack")
    combined = sim.query_metric_history(
        0,
        3,
        entity_id="rack-r1-row1-01",
        entity_type="rack",
        metric_name="rack_power_kw",
    )

    assert by_metric
    assert {series.metric_name for series in by_metric} == {"facility_power_kw"}
    assert by_entity
    assert {series.entity_id for series in by_entity} == {"datacenter"}
    assert by_type
    assert {series.entity_type for series in by_type} == {"rack"}
    assert len(combined) == 1
    assert combined[0].entity_id == "rack-r1-row1-01"
    assert combined[0].metric_name == "rack_power_kw"


def test_every_metric_history_point_has_cataloged_entity_unit_and_kind():
    sim = make_history_sim()
    sim.step(1)
    catalog_names = {record["metric_name"] for record in metric_catalog_records()}

    payload = dump_series(sim.query_metric_history(0, 2))

    assert payload
    for series in payload:
        assert series["metric_name"] in catalog_names
        assert series["entity_id"]
        assert series["entity_type"]
        assert series["unit"]
        assert series["metric_kind"] in {"counter", "gauge"}
        for point in series["points"]:
            assert set(point) == {"sim_time_seconds", "value"}


def test_identical_episodes_produce_identical_metric_history_hashes():
    first = make_history_sim()
    second = make_history_sim()
    first.step(3)
    second.step(3)

    assert history_hash(first.query_metric_history(0, 4)) == history_hash(
        second.query_metric_history(0, 4)
    )


def test_metric_history_is_agent_safe_and_excludes_evaluator_only_fields():
    sim = make_history_sim()
    sim.step(1)
    payload = {"series": dump_series(sim.query_metric_history(0, 2))}
    serialized = json.dumps(payload, sort_keys=True).lower()

    assert_no_agent_leakage(payload)
    for forbidden in (
        "active_faults",
        "fault_type",
        "fault_target",
        "score_hints",
        "seed",
        "baseline_capacity_kw",
        "desired_allocated_server_ids",
        "desired_allocation_weights",
    ):
        assert forbidden not in serialized


def test_agent_metric_history_uses_canonical_rack_visibility_not_raw_host_store(
    monkeypatch,
):
    sim = make_history_sim()
    sim.step(1)
    raw_series = sim.query_metric_history(0, 2)

    assert any(series.entity_type == "server" for series in raw_series)
    monkeypatch.setattr(
        sim,
        "query_metric_history",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("agent endpoint accessed raw metric history")
        ),
    )

    response = agent_metric_history(sim, 0, 2)
    serialized = json.dumps(response, sort_keys=True)

    assert response["series"]
    assert all(series["entity_type"] != "server" for series in response["series"])
    assert not re.search(r"server-r\d+-row\d+-rack\d+-\d+", serialized)
    assert agent_metric_history(sim, 0, 2, entity_type="server") == {"series": []}
    assert agent_metric_history(
        sim,
        0,
        2,
        entity_id="server-r1-row1-rack01-01",
    ) == {"series": []}
    assert_no_agent_leakage(response)


def test_agent_metric_history_matches_canonical_monitoring_delivery():
    sim = DataCenterSimulator()
    sim.reset(seed=7)
    sim.start_workload(
        {
            "request_rate_per_second": 300,
            "placement_strategy": "spread",
            "noise_enabled": False,
        }
    )
    sim.step(5)
    sim.inject_fault(
        FaultRequest(
            fault_type="monitoring_pipeline_failure",
            target="monitoring-pipeline",
            severity=0.85,
            duration_seconds=300,
        )
    )
    sim.step(3)
    end_time = sim.sim_time_seconds + 1

    canonical = sim.canonical_snapshot(
        lookback_seconds=sim.sim_time_seconds,
        channels={"metric"},
        include_config=False,
    )
    canonical_latency = next(
        observation
        for observation in canonical["observations"]
        if observation["payload"]["metric_name"] == "workload.average_latency"
    )
    expected_points = [
        {"sim_time_seconds": int(timestamp), "value": float(value)}
        for timestamp, value, missing in zip(
            canonical_latency["payload"]["timestamps_seconds"],
            canonical_latency["payload"]["values"],
            canonical_latency["payload"]["missingness_mask"],
            strict=True,
        )
        if not missing and value is not None
    ]

    response = agent_metric_history(
        sim,
        0,
        end_time,
        metric_name="workload_average_latency_ms",
    )
    raw_points = all_points(
        sim.query_metric_history(
            0,
            end_time,
            metric_name="workload_average_latency_ms",
        )
    )

    assert response["series"][0]["points"] == expected_points
    assert len(expected_points) < len(raw_points)
    assert any(canonical_latency["payload"]["missingness_mask"])
