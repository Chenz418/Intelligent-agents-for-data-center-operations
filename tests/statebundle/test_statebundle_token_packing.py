"""Regression tests for exact compact StateBundle bundle packing."""

from __future__ import annotations

from typing import Any

from aiopslab.agent_telemetry import (
    AgentObservationRequest,
    compact_statebundle_bundle_token_cost,
    compact_statebundle_observation_token_cost,
    render_snapshot,
)


QUERY_TIME = 100.0


def _config(observation_id: str, value: float) -> dict[str, Any]:
    return {
        "observation_id": observation_id,
        "channel": "config",
        "window": {
            "start_time_seconds": QUERY_TIME,
            "end_time_seconds": QUERY_TIME,
            "start_inclusive": True,
            "end_inclusive": True,
        },
        "payload": {
            "unit_type": "configuration_state",
            "scope": "rack/rack-a",
            "path": f"cooling.setting.{observation_id}",
            "operation": "set",
            "value_type": "number",
            "previous_value": value - 1.0,
            "value": value,
            "change_time_seconds": QUERY_TIME,
        },
        "metadata": {
            "event_start_time_seconds": QUERY_TIME,
            "event_end_time_seconds": QUERY_TIME,
            "ingest_time_seconds": QUERY_TIME,
            "available_at_time_seconds": QUERY_TIME,
            "available_at_sequence": "cut-packing",
            "entities": [
                {
                    "entity_id": "rack-a",
                    "role": "scope",
                    "confidence": 1.0,
                    "provenance": "inventory",
                }
            ],
            "primary_subsystem": "cooling",
            "primary_subsystem_provenance": "inventory",
            "correlation_ids": {},
            "source_references": [f"source://{observation_id}"],
            "data_quality": {
                "parse_confidence": 1.0,
                "missingness_fraction": 0.0,
                "delay_seconds": 0.0,
                "availability_mask": {"payload": True},
                "validation_flags": [],
            },
        },
    }


def _group_references(observations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    references = [str(item["observation_id"]) for item in observations]
    return [{"group": 0, "anchor": references[0], "supports": references[1:]}]


def _statebundle(
    observations: list[dict[str, Any]], token_budget: int
) -> dict[str, Any]:
    return {
        "schema_version": "statebundle.output.v1",
        "incident_id": "episode-packing",
        "snapshot_id": "snapshot-packing",
        "query_time_seconds": QUERY_TIME,
        "token_budget": token_budget,
        "used_tokens": token_budget,
        "candidate_count": len(observations),
        "evidence_groups": [
            {
                "anchor": observations[0],
                "corroborating_observations": observations[1:],
            }
        ],
    }


def _canonical_context(observations: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": "statebundle.canonical.v1",
        "episode_id": "episode-packing",
        "snapshot_id": "snapshot-packing",
        "query_time_seconds": QUERY_TIME,
        "query_watermark_sequence": "cut-packing",
        "observations": observations,
    }


def _represented_config_ids(rendered: dict[str, Any]) -> list[str]:
    table = rendered["tables"]["config"]
    id_index = table["columns"].index("id")
    return [str(row[id_index]) for row in table["rows"]]


def test_shared_bundle_envelopes_admit_row_rejected_by_singleton_sum() -> None:
    observations = [_config("config-a", 21.0), _config("config-b", 22.0)]
    request = AgentObservationRequest(channels=("config",))
    singleton_costs = [
        compact_statebundle_observation_token_cost(
            observation,
            query_time_seconds=QUERY_TIME,
            request=request,
        )
        for observation in observations
    ]
    exact_bundle_cost = compact_statebundle_bundle_token_cost(
        observations,
        _group_references(observations),
        query_time_seconds=QUERY_TIME,
        request=request,
    )

    assert singleton_costs[0] <= exact_bundle_cost < sum(singleton_costs)

    rendered = render_snapshot(
        _statebundle(observations, exact_bundle_cost),
        canonical_context=_canonical_context(observations),
        condition="statebundle",
        request=request,
    )

    assert _represented_config_ids(rendered) == ["config-a", "config-b"]
    assert rendered["budget"]["serialization_admission_dropped_count"] == 0
    assert (
        rendered["budget"]["serialization_admission_used_tokens"] == exact_bundle_cost
    )
    assert rendered["budget"]["actual_serialized_tokens"] == exact_bundle_cost
    assert (
        rendered["budget"]["actual_serialized_tokens"]
        <= rendered["budget"]["token_budget"]
    )


def test_marginal_bundle_admission_never_exceeds_budget() -> None:
    observations = [
        _config("config-a", 21.0),
        _config("config-b", 22.0),
        _config("config-c", 23.0),
    ]
    request = AgentObservationRequest(channels=("config",))
    admitted = observations[:2]
    token_budget = compact_statebundle_bundle_token_cost(
        admitted,
        _group_references(admitted),
        query_time_seconds=QUERY_TIME,
        request=request,
    )
    all_rows_cost = compact_statebundle_bundle_token_cost(
        observations,
        _group_references(observations),
        query_time_seconds=QUERY_TIME,
        request=request,
    )
    assert all_rows_cost > token_budget

    rendered = render_snapshot(
        _statebundle(observations, token_budget),
        canonical_context=_canonical_context(observations),
        condition="statebundle",
        request=request,
    )

    assert _represented_config_ids(rendered) == ["config-a", "config-b"]
    assert rendered["budget"]["serialization_admission_dropped_count"] == 1
    assert (
        rendered["budget"]["serialization_admission_used_tokens"]
        == rendered["budget"]["actual_serialized_tokens"]
    )
    assert rendered["budget"]["actual_serialized_tokens"] <= token_budget


def test_empty_evidence_has_zero_telemetry_budget_cost() -> None:
    assert (
        compact_statebundle_bundle_token_cost(
            [],
            [],
            query_time_seconds=QUERY_TIME,
            request=AgentObservationRequest(),
        )
        == 0
    )
