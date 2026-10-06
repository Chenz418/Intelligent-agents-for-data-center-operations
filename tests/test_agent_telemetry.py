"""Focused contracts for deterministic agent-facing telemetry rendering.

These tests intentionally use small canonical-shaped dictionaries.  The
canonical validator and StateBundle schema tests live elsewhere; this suite
guards only the projection boundary and therefore stays fast.
"""

from __future__ import annotations

from copy import deepcopy
import json
from typing import Any

import pytest

from aiopslab.agent_telemetry import (
    AGENT_DELTA_SCHEMA_VERSION,
    AGENT_SCHEMA_VERSION,
    AgentObservationRequest,
    AgentTelemetryRenderer,
    compact_statebundle_bundle_token_cost,
    compact_statebundle_observation_token_cost,
    logical_observation_key,
    render_snapshot,
)


QUERY_TIME = 100.0
CAUSAL_CUT = "cut-000042"


def _quality(**overrides: Any) -> dict[str, Any]:
    quality: dict[str, Any] = {
        "parse_confidence": 1.0,
        "missingness_fraction": 0.0,
        "delay_seconds": 0.0,
        "availability_mask": {"payload": True, "window": True},
        "validation_flags": [],
    }
    quality.update(overrides)
    return quality


def _observation(
    channel: str,
    observation_id: str,
    payload: dict[str, Any],
    *,
    start: float = 40.0,
    end: float = QUERY_TIME,
    subsystem: str = "cooling",
    entity_id: str = "rack-a",
    quality: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "observation_id": observation_id,
        "channel": channel,
        "window": {
            "start_time_seconds": start,
            "end_time_seconds": end,
            "start_inclusive": True,
            "end_inclusive": True,
        },
        "payload": payload,
        "metadata": {
            "event_start_time_seconds": start,
            "event_end_time_seconds": end,
            "ingest_time_seconds": end,
            "available_at_time_seconds": end,
            "available_at_sequence": CAUSAL_CUT,
            "entities": [
                {
                    "entity_id": entity_id,
                    "role": "producer",
                    "confidence": 1.0,
                    "provenance": f"entity-provenance-secret-{observation_id}",
                }
            ],
            "primary_subsystem": subsystem,
            "primary_subsystem_provenance": (
                f"subsystem-provenance-secret-{observation_id}"
            ),
            "correlation_ids": {"trace_id": f"private-trace-{observation_id}"},
            "source_references": [f"source-secret://{observation_id}"],
            "data_quality": quality or _quality(),
        },
    }


def _metric(
    observation_id: str = "metric-1",
    *,
    end: float = QUERY_TIME,
    entity_id: str = "rack-a",
    timestamps: list[float] | None = None,
    values: list[float | None] | None = None,
    missingness_mask: list[bool] | None = None,
    quality: dict[str, Any] | None = None,
) -> dict[str, Any]:
    timestamps = timestamps or [40, 50, 60, 70, 80, 90, 100]
    values = values or [1.0, 2.0, 3.0, 4.0, None, 6.0, 7.0]
    missingness_mask = missingness_mask or [
        False,
        False,
        False,
        False,
        True,
        False,
        False,
    ]
    numeric = [value for value in values if value is not None]
    payload = {
        "unit_type": "metric_series",
        "metric_name": "inlet_temperature_celsius",
        "resource": {"entity_id": entity_id, "entity_type": "rack"},
        "unit": "celsius",
        "scale": "linear",
        "sample_period_seconds": 10.0,
        "timestamps_seconds": timestamps,
        "values": values,
        "missingness_mask": missingness_mask,
        "statistics": {
            "count": len(numeric),
            "last": numeric[-1],
            "max": max(numeric),
            "mean": sum(numeric) / len(numeric),
            "median": 3.5,
            "min": min(numeric),
            "p95": 7.0,
            "slope_per_second": 0.1,
            "stddev": 2.0,
        },
        "normalization_reference": {
            "method": "robust",
            "sample_count": len(numeric),
            "median": 3.5,
            "iqr": 3.5,
            "last": numeric[-1],
            "reference_end_time_seconds": end,
        },
    }
    return _observation(
        "metric",
        observation_id,
        payload,
        start=float(timestamps[0]),
        end=end,
        entity_id=entity_id,
        quality=quality,
    )


def _log(
    observation_id: str = "log-1",
    *,
    end: float = QUERY_TIME,
    variables: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return _observation(
        "log",
        observation_id,
        {
            "unit_type": "log_template",
            "template_id": f"template-{observation_id}",
            "template": "fan controller reported speed {rpm}",
            "event_type": "controller",
            "count": 4,
            "severity": "warning",
            "severity_histogram": {"warning": 4},
            "rarity": 0.1,
            "burst_rate_per_minute": 2.0,
            "variable_summaries": variables or {"rpm": {"last": 800}},
            "time_features": {"last_seen_seconds": end},
        },
        end=end,
    )


def _alert(
    observation_id: str,
    target: str,
    *,
    quality: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return _observation(
        "alert",
        observation_id,
        {
            "unit_type": "alert_event",
            "alert_type": "RackFanSlow",
            "alert_fingerprint": f"rack-fan-slow:{target}",
            "message": f"Fan alarm on {target}",
            "target": target,
            "status": "firing",
            "severity": "warning",
            "threshold": 900,
            "duration_seconds": 15.0,
            "details": {"rack": target, "limit": 900},
        },
        entity_id=target,
        quality=quality,
    )


def _trace(observation_id: str = "trace-1") -> dict[str, Any]:
    return _observation(
        "trace",
        observation_id,
        {
            "unit_type": "trace_aggregate",
            "operation": "schedule_workload",
            "source": "scheduler",
            "destination": "rack-a",
            "count": 3,
            "status_counts": {"ok": 2, "error": 1},
            "retry_count": 1,
            "latency_ms": {"median": 12.0, "p95": 20.0},
            "critical_path": ["scheduler", "rack-a"],
        },
    )


def _config(
    observation_id: str = "config-1",
    *,
    end: float = QUERY_TIME,
) -> dict[str, Any]:
    return _observation(
        "config",
        observation_id,
        {
            "unit_type": "configuration_state",
            "scope": "rack/rack-a",
            "path": "cooling.target_temperature_celsius",
            "operation": "set",
            "value_type": "number",
            "previous_value": 24.0,
            "value": 22.0,
            "change_time_seconds": end - 5,
        },
        end=end,
    )


def _snapshot(
    observations: list[dict[str, Any]],
    *,
    query_time: float = QUERY_TIME,
    snapshot_id: str = "snapshot-1",
    causal_cut: str = CAUSAL_CUT,
) -> dict[str, Any]:
    return {
        "schema_version": "statebundle.canonical.v1",
        "episode_id": "episode-renderer-test",
        "snapshot_id": snapshot_id,
        "query_time_seconds": query_time,
        "query_watermark_sequence": causal_cut,
        "channels": ["log", "metric", "alert", "trace", "config"],
        "observations": observations,
    }


def _row(table: dict[str, Any], index: int = 0) -> dict[str, Any]:
    return dict(zip(table["columns"], table["rows"][index], strict=True))


def _represented_ids(rendered: dict[str, Any]) -> set[str]:
    represented: set[str] = set()
    for channel, table in rendered["tables"].items():
        if channel == "alert":
            members_index = table["columns"].index("members")
            member_id_index = rendered["table_schema"]["alert_member_columns"].index(
                "id"
            )
            represented.update(
                member[member_id_index]
                for row in table["rows"]
                for member in row[members_index]
            )
        else:
            id_index = table["columns"].index("id")
            represented.update(row[id_index] for row in table["rows"])
    return represented


def test_full_compact_is_deterministic_complete_and_hides_repeated_metadata() -> None:
    observations = [
        _log(),
        _metric(
            "metric-nondefault-quality",
            entity_id="rack-metric",
            quality=_quality(
                parse_confidence=0.75,
                missingness_fraction=0.25,
                delay_seconds=3.0,
                availability_mask={"payload": True, "window": False},
                validation_flags=["late_sample"],
            ),
        ),
        _alert("alert-a", "rack-a"),
        _alert(
            "alert-b",
            "rack-b",
            quality=_quality(delay_seconds=2.0, validation_flags=["delayed"]),
        ),
        _trace(),
        _config(),
    ]
    canonical = _snapshot(observations)
    canonical_before = deepcopy(canonical)

    first = render_snapshot(canonical)
    second = render_snapshot(deepcopy(canonical))

    assert first == second
    assert canonical == canonical_before
    assert first["schema_version"] == AGENT_SCHEMA_VERSION
    assert first["causal_cut"] == CAUSAL_CUT
    assert first["channel_counts"] == {
        "log": 1,
        "metric": 1,
        "alert": 2,
        "trace": 1,
        "config": 1,
    }
    assert _represented_ids(first) == {
        observation["observation_id"] for observation in observations
    }

    # Both alerts survive even though their structurally repeated fields share
    # one row.  Target-specific fields remain in reconstructable members.
    alert_table = first["tables"]["alert"]
    assert len(alert_table["rows"]) == 1
    alert_row = _row(alert_table)
    assert alert_row["message"] == "Fan alarm on {target}"
    assert alert_row["details"] == {"limit": 900, "rack": "{target}"}
    member_columns = first["table_schema"]["alert_member_columns"]
    members = [
        dict(zip(member_columns, row, strict=True)) for row in alert_row["members"]
    ]
    assert {
        (member["id"], member["target"], member["fingerprint"]) for member in members
    } == {
        ("alert-a", "rack-a", "rack-fan-slow:rack-a"),
        ("alert-b", "rack-b", "rack-fan-slow:rack-b"),
    }
    assert all(member["duration"] == 15.0 for member in members)
    assert {member["target"]: member["entities"] for member in members} == {
        "rack-a": [["rack-a", "producer"]],
        "rack-b": [["rack-b", "producer"]],
    }

    metric_row = _row(first["tables"]["metric"])
    assert metric_row["entity"] == ["rack-metric", "rack"]
    assert metric_row["entities"] is None  # no duplicated resource entity
    assert metric_row["quality"] == {
        "parse_confidence": 0.75,
        "missingness_fraction": 0.25,
        "delay_seconds": 3.0,
        "unavailable": ["window"],
        "validation_flags": ["late_sample"],
    }
    assert _row(first["tables"]["log"])["quality"] is None
    assert members[0]["quality"] is None
    assert members[1]["quality"] == {
        "delay_seconds": 2.0,
        "validation_flags": ["delayed"],
    }

    # Provenance/source/correlation fields remain canonical-only.  The cut is
    # represented once at snapshot level rather than once per row.
    serialized = json.dumps(first, sort_keys=True)
    assert serialized.count(CAUSAL_CUT) == 1
    assert "source-secret://" not in serialized
    assert "private-trace-" not in serialized
    assert "provenance-secret" not in serialized
    for hidden_key in (
        "source_references",
        "correlation_ids",
        "primary_subsystem_provenance",
        "available_at_sequence",
        "ingest_time_seconds",
    ):
        assert f'"{hidden_key}"' not in serialized

    config_row = _row(first["tables"]["config"])
    assert config_row["scope"] == "rack/rack-a"
    assert config_row["path"] == "cooling.target_temperature_celsius"
    assert config_row["operation"] == "set"
    assert config_row["previous_value"] == 24.0
    assert config_row["value"] == 22.0
    assert config_row["change_time"] == 95.0


def test_metric_overview_is_multiscale_and_raw_drilldown_is_exact() -> None:
    metric = _metric()
    canonical = _snapshot([metric])

    overview = render_snapshot(
        canonical,
        request=AgentObservationRequest(channels=("metric",), lookback_seconds=100),
        recent_seconds=30,
    )
    metric_table = overview["tables"]["metric"]
    metric_row = _row(metric_table)
    recent_columns = overview["table_schema"]["metric_recent_columns"]
    recent = dict(zip(recent_columns, metric_row["recent"], strict=True))
    assert metric_table["recent_time_axes"][recent["axis"]] == [70.0, 80.0, 90.0, 100.0]
    assert recent["values"] == [4.0, None, 6.0, 7.0]
    assert recent["missing_indices"] == [1]

    baseline_columns = overview["table_schema"]["metric_baseline_columns"]
    baseline = dict(zip(baseline_columns, metric_row["baseline"], strict=True))
    assert baseline == {
        "start": 40.0,
        "end": 60.0,
        "sample_count": 3,
        "observed_count": 3,
        "median": 2.0,
        "q1": 1.5,
        "q3": 2.5,
        "last": 3.0,
        "trend_per_second": pytest.approx(0.1),
        "change_points": [],
    }
    assert metric_row["statistics"] is not None
    assert metric_row["normalization"] is not None

    raw = render_snapshot(
        canonical,
        request=AgentObservationRequest(
            channels=("metric",),
            lookback_seconds=100,
            detail="raw",
            metric_names=("inlet_temperature_celsius",),
            entity_ids=("rack-a",),
        ),
    )
    raw_row = _row(raw["tables"]["metric"])
    assert raw["table_schema"]["metric_mode"] == "raw_series"
    assert raw_row["timestamps"] == metric["payload"]["timestamps_seconds"]
    assert raw_row["values"] == metric["payload"]["values"]
    assert raw_row["missingness_mask"] == metric["payload"]["missingness_mask"]


def test_short_lookback_recomputes_metric_slice_and_keeps_config_state() -> None:
    rendered = render_snapshot(
        _snapshot([_metric(), _config()]),
        request=AgentObservationRequest(lookback_seconds=20, detail="raw"),
    )

    metric_row = _row(rendered["tables"]["metric"])
    assert metric_row["timestamps"] == [80.0, 90.0, 100.0]
    statistic_columns = rendered["table_schema"]["metric_statistic_columns"]
    metric_statistics = dict(
        zip(statistic_columns, metric_row["statistics"], strict=True)
    )
    assert metric_statistics["count"] == 2.0
    assert metric_statistics["mean"] == 6.5
    normalization_columns = rendered["table_schema"]["metric_normalization_columns"]
    normalization = dict(
        zip(normalization_columns, metric_row["normalization"], strict=True)
    )
    assert normalization == {
        "method": "causal_pre_window_robust",
        "sample_count": 4,
        "median": 2.5,
        "iqr": 1.5,
        "last": 4.0,
        "reference_end_time_seconds": 70.0,
    }
    assert _row(rendered["tables"]["config"])["id"] == "config-1"


def test_log_variable_summaries_are_capped_in_overview_and_restored_in_raw() -> None:
    variables = {
        f"variable-{index:02d}": {"last": index, "count": index + 1}
        for index in range(15)
    }
    canonical = _snapshot([_log(variables=variables)])

    overview = render_snapshot(
        canonical,
        request=AgentObservationRequest(channels=("log",)),
    )
    overview_row = _row(overview["tables"]["log"])
    assert overview_row["variable_count"] == 15
    assert list(overview_row["variables"]) == sorted(variables)[:12]
    assert len(overview_row["variables"]) == 12

    raw = render_snapshot(
        canonical,
        request=AgentObservationRequest(channels=("log",), detail="raw"),
    )
    raw_row = _row(raw["tables"]["log"])
    assert raw_row["variable_count"] == 15
    assert raw_row["variables"] == dict(sorted(variables.items()))


def test_delta_reports_lifecycle_quality_and_only_appended_metric_points() -> None:
    original_metric = _metric(
        "metric-v1",
        timestamps=[70, 80, 90, 100],
        values=[4.0, 5.0, 6.0, 7.0],
        missingness_mask=[False, False, False, False],
    )
    initial = _snapshot(
        [original_metric, _config("config-removed")],
        snapshot_id="snapshot-before",
        causal_cut="cut-before",
    )
    updated_metric = _metric(
        "metric-v2",
        end=110,
        timestamps=[70, 80, 90, 100, 110],
        values=[4.0, 5.0, 6.0, 7.0, 8.0],
        missingness_mask=[False, False, False, False, False],
        quality=_quality(parse_confidence=0.8, validation_flags=["partial_decode"]),
    )
    current = _snapshot(
        [updated_metric, _log("log-added", end=110)],
        query_time=110,
        snapshot_id="snapshot-after",
        causal_cut="cut-after",
    )
    renderer = AgentTelemetryRenderer(recent_seconds=30)

    first = renderer.render(initial, initial=True)
    delta = renderer.render(current)

    assert first["schema_version"] == AGENT_SCHEMA_VERSION
    assert delta["schema_version"] == AGENT_DELTA_SCHEMA_VERSION
    assert delta["base_snapshot_id"] == "snapshot-before"
    assert delta["snapshot_id"] == "snapshot-after"
    assert delta["causal_cut"] == "cut-after"
    assert _row(delta["added"]["log"])["id"] == "log-added"
    assert delta["removed"] == [
        {
            "key": logical_observation_key(_config("config-removed")),
            "channel": "config",
            "observation_ids": ["config-removed"],
        }
    ]

    updated_table = delta["updated"]["metric"]
    updated_row = _row(updated_table)
    # Delta rows reuse the static cell schema from the initial snapshot.
    assert "table_schema" not in delta
    recent_columns = first["table_schema"]["metric_recent_columns"]
    recent = dict(zip(recent_columns, updated_row["recent"], strict=True))
    assert updated_table["recent_time_axes"][recent["axis"]] == [110.0]
    assert recent["values"] == [8.0]
    assert recent["missing_indices"] is None
    assert updated_row["quality"] == {
        "parse_confidence": 0.8,
        "validation_flags": ["partial_decode"],
    }
    assert delta["quality_changes"] == [
        {
            "key": logical_observation_key(updated_metric),
            "after": {
                "parse_confidence": 0.8,
                "validation_flags": ["partial_decode"],
            },
        }
    ]


def test_delta_ignores_hidden_cut_changes_and_emits_metric_corrections() -> None:
    config_before = _config()
    metric_before = _metric(
        "metric-stable",
        timestamps=[80, 90, 100],
        values=[2.0, 3.0, 4.0],
        missingness_mask=[False, False, False],
    )
    first_snapshot = _snapshot(
        [config_before, metric_before],
        snapshot_id="snapshot-before",
        causal_cut="cut-before",
    )
    config_after = deepcopy(config_before)
    config_after["metadata"]["available_at_sequence"] = "cut-after"
    metric_after = deepcopy(metric_before)
    metric_after["payload"]["values"] = [2.0, 30.0, 4.0]
    metric_after["metadata"]["available_at_sequence"] = "cut-after"
    second_snapshot = _snapshot(
        [config_after, metric_after],
        snapshot_id="snapshot-after",
        causal_cut="cut-after",
    )
    renderer = AgentTelemetryRenderer()

    initial = renderer.render(first_snapshot, initial=True)
    delta = renderer.render(second_snapshot)

    assert delta["updated"].get("config") is None
    metric_row = _row(delta["updated"]["metric"])
    recent_columns = initial["table_schema"]["metric_recent_columns"]
    recent = dict(zip(recent_columns, metric_row["recent"], strict=True))
    assert delta["updated"]["metric"]["recent_time_axes"][recent["axis"]] == [90.0]
    assert recent["values"] == [30.0]


def test_delta_includes_schema_when_nested_channel_first_appears() -> None:
    renderer = AgentTelemetryRenderer()
    renderer.render(_snapshot([_config()]), initial=True)

    delta = renderer.render(
        _snapshot([_config(), _alert("alert-new", "rack-a")], snapshot_id="next")
    )

    assert delta["table_schema"]["alert_member_columns"]


def test_statebundle_drilldown_does_not_leak_filtered_group_keys() -> None:
    visible = _metric("metric-visible", entity_id="rack-a")
    hidden = _metric("metric-hidden", entity_id="rack-b")
    canonical = _snapshot([visible, hidden])
    statebundle = {
        "schema_version": "statebundle.output.v1",
        "incident_id": canonical["episode_id"],
        "snapshot_id": canonical["snapshot_id"],
        "query_time_seconds": QUERY_TIME,
        "token_budget": 4096,
        "used_tokens": 100,
        "candidate_count": 2,
        "evidence_groups": [
            {
                "anchor": visible,
                "corroborating_observations": [hidden],
            }
        ],
    }

    rendered = render_snapshot(
        statebundle,
        canonical_context=canonical,
        condition="statebundle",
        request=AgentObservationRequest(
            channels=("metric",),
            entity_ids=("rack-a",),
            detail="raw",
        ),
    )

    serialized_groups = json.dumps(rendered["evidence_groups"], sort_keys=True)
    assert "metric-visible" in serialized_groups
    assert "metric-hidden" not in serialized_groups


def test_statebundle_target_scope_summary_is_budgeted_and_filtered_with_evidence() -> (
    None
):
    direct = _metric("metric-direct", entity_id="rack-a")
    downstream = _metric("metric-downstream", entity_id="rack-b")
    canonical = _snapshot([direct, downstream])
    groups = [
        {
            "group": 0,
            "anchor": "metric-direct",
            "supports": ["metric-downstream"],
        }
    ]
    scope_candidates = [
        {
            "scope": "control-plane",
            "estimated_role": "direct_target_candidate",
            "supporting_observation_ids": ["metric-direct"],
        },
        {
            "scope": "workload",
            "estimated_role": "downstream_affected_scope",
            "supporting_observation_ids": ["metric-downstream"],
        },
    ]
    legacy_cost = compact_statebundle_bundle_token_cost(
        [direct, downstream],
        groups,
        query_time_seconds=QUERY_TIME,
    )
    exact_cost = compact_statebundle_bundle_token_cost(
        [direct, downstream],
        groups,
        query_time_seconds=QUERY_TIME,
        target_scope_ambiguity=True,
        target_scope_candidates=scope_candidates,
    )
    assert exact_cost > legacy_cost
    statebundle = {
        "schema_version": "statebundle.output.v1",
        "incident_id": canonical["episode_id"],
        "snapshot_id": canonical["snapshot_id"],
        "query_time_seconds": QUERY_TIME,
        "token_budget": exact_cost,
        "used_tokens": exact_cost,
        "candidate_count": 2,
        "target_scope_ambiguity": True,
        "target_scope_candidates": scope_candidates,
        "evidence_groups": [
            {
                "anchor": direct,
                "corroborating_observations": [downstream],
            }
        ],
    }

    rendered = render_snapshot(
        statebundle,
        canonical_context=canonical,
        condition="statebundle",
    )

    assert rendered["budget"]["actual_serialized_tokens"] == exact_cost
    assert rendered["budget"]["within_budget"] is True
    assert rendered["target_scope_ambiguity"] is True
    assert rendered["target_scope_candidates"] == scope_candidates

    targeted = render_snapshot(
        statebundle,
        canonical_context=canonical,
        condition="statebundle",
        request=AgentObservationRequest(
            channels=("metric",),
            entity_ids=("rack-a",),
        ),
    )

    assert targeted["target_scope_ambiguity"] is False
    assert targeted["target_scope_candidates"] == [scope_candidates[0]]
    serialized_scope = json.dumps(targeted["target_scope_candidates"], sort_keys=True)
    assert "metric-direct" in serialized_scope
    assert "metric-downstream" not in serialized_scope
    assert targeted["budget"]["within_budget"] is True


def test_budgeted_raw_drilldown_recosts_actual_series_without_failure() -> None:
    timestamps = list(range(301))
    metric = _metric(
        "metric-long-raw",
        end=300,
        timestamps=timestamps,
        values=[float(value) for value in timestamps],
        missingness_mask=[False] * len(timestamps),
    )
    overview_request = AgentObservationRequest(channels=("metric",))
    raw_request = AgentObservationRequest(channels=("metric",), detail="raw")
    statebundle_budget = (
        compact_statebundle_observation_token_cost(
            metric,
            query_time_seconds=300,
            request=overview_request,
        )
        + 5
    )
    statebundle = {
        "schema_version": "statebundle.output.v1",
        "incident_id": "episode-renderer-test",
        "snapshot_id": "snapshot-raw",
        "query_time_seconds": 300,
        "token_budget": statebundle_budget,
        "used_tokens": statebundle_budget - 1,
        "evidence_groups": [{"anchor": metric, "corroborating_observations": []}],
    }
    selected = render_snapshot(
        statebundle,
        canonical_context=_snapshot([metric], query_time=300),
        condition="statebundle",
        request=raw_request,
    )

    assert selected["budget"]["actual_serialized_tokens"] <= statebundle_budget
    assert selected["tables"] == {}
    assert selected["evidence_groups"] == []
