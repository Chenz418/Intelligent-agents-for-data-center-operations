"""StateBundle inference-boundary and budgeted-retrieval tests."""

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import Tensor, nn

from aiopslab.statebundle.config import StateBundleInferenceConfig
from aiopslab.statebundle.data import (
    parse_canonical_snapshot,
)
from aiopslab.statebundle.inference import ProjectedANNIndex, StateBundleInference
from aiopslab.statebundle.runtime import _normalize_action_context
from aiopslab.statebundle.types import (
    CANONICAL_SCHEMA_VERSION,
    CanonicalIncident,
    CanonicalObservation,
    IncidentBatch,
    TelemetryChannel,
)


CHANNELS = tuple(channel.value for channel in TelemetryChannel)
FORBIDDEN_AGENT_KEYS = {
    "annotations",
    "associated_effect_ids",
    "associated_fault_effect_ids",
    "background",
    "causal_role",
    "effect_id",
    "embedding",
    "embeddings",
    "fault_id",
    "fault_label_id",
    "fault_mechanism",
    "fault_target",
    "gate",
    "ground_truth",
    "hidden_training_annotations",
    "inclusion_probabilities",
    "local_effect_id",
    "local_effect_or_anomaly_id",
    "pair_label",
    "phase",
    "propagation_depth",
    "propagation_membership",
    "propagation_path",
    "prototype",
    "prototype_id",
    "relevance_score",
    "root_cause",
    "root_cause_label",
    "selector_gate",
    "source_references",
    "subsystem_supervision",
    "symptom_family",
    "training_annotations",
    "training_labels",
}


def _observation_dict(
    channel: str,
    index: int,
    *,
    start: float = 1.0,
    end: float = 2.0,
    available: float = 2.0,
    watermark: str = "cut-query-1",
    source_reference: str | None = None,
) -> dict[str, Any]:
    return {
        "observation_id": f"obs-{index:03d}",
        "channel": channel,
        "window": {
            "start_time_seconds": start,
            "end_time_seconds": end,
            "start_inclusive": True,
            "end_inclusive": True,
        },
        "payload": {
            "unit_type": f"{channel}_unit",
            "message": f"observable-{channel}-{index}",
            "ordinal": index,
        },
        "metadata": {
            "event_start_time_seconds": start,
            "event_end_time_seconds": end,
            "ingest_time_seconds": available,
            "available_at_time_seconds": available,
            "available_at_sequence": watermark,
            "entities": [
                {
                    "entity_id": f"rack-{index % 2}",
                    "role": "affected" if index == 0 else "producer",
                    "confidence": 0.9,
                    "provenance": "inventory",
                }
            ],
            "primary_subsystem": f"subsystem-{index % 2}",
            "primary_subsystem_provenance": "inventory",
            "correlation_ids": {
                "trace_id": f"private-correlation-{index}",
            },
            "source_references": [source_reference or f"private-source-{index}"],
            "data_quality": {
                "parse_confidence": 0.95,
                "missingness_fraction": 0.0,
                "delay_seconds": max(0.0, available - end),
                "availability_mask": {"payload": True},
                "validation_flags": [],
            },
        },
    }


def _snapshot(
    observations: list[dict[str, Any]] | None = None,
    *,
    query_time: float = 10.0,
    watermark: str = "cut-query-1",
) -> dict[str, Any]:
    if observations is None:
        observations = [
            _observation_dict(channel, index, watermark=watermark)
            for index, channel in enumerate(CHANNELS)
        ]
    return {
        "schema_version": CANONICAL_SCHEMA_VERSION,
        "episode_id": "incident-visible-1",
        "snapshot_id": "snapshot-visible-1",
        "query_time_seconds": query_time,
        "query_watermark_sequence": watermark,
        "observations": observations,
    }


def _incident(observations: list[dict[str, Any]] | None = None) -> CanonicalIncident:
    return CanonicalIncident.from_dict(_snapshot(observations))


class _DeterministicModel(nn.Module):
    """Minimal observable-only model surface used to isolate retrieval tests."""

    def __init__(self, relevance: list[float], aspects: list[list[float]]):
        super().__init__()
        self._relevance: Tensor
        self._aspects: Tensor
        self.register_buffer("_relevance", torch.tensor(relevance, dtype=torch.float32))
        self.register_buffer("_aspects", torch.tensor(aspects, dtype=torch.float32))
        self.seen_batch_types: list[type[Any]] = []

    def forward(self, batch: IncidentBatch) -> SimpleNamespace:
        if not isinstance(batch, IncidentBatch):
            raise AssertionError("inference model received a non-observable batch")
        self.seen_batch_types.append(type(batch))
        count = len(batch.incidents[0].observations)
        if count != self._relevance.numel() or count != self._aspects.shape[0]:
            raise AssertionError("fixture/model cardinality mismatch")
        return SimpleNamespace(
            relevance_scores=self._relevance.unsqueeze(0),
            aspect_embeddings=self._aspects.unsqueeze(0),
        )


class _UnitTokenEstimator:
    def estimate(self, observation: CanonicalObservation, policy: Any) -> int:
        del observation, policy
        return 1


def _service(
    relevance: list[float],
    aspects: list[list[float]],
    **config_overrides: Any,
) -> StateBundleInference:
    defaults: dict[str, Any] = {
        "top_m": len(relevance),
        "anchor_token_budget": 2,
        "total_token_budget": 4,
        "max_anchors": 2,
        "supports_per_anchor": 1,
        "diversity_weight": 0.0,
        "minimum_anchor_gain": 0.0,
        "ann_projection_count": 2,
        "ann_candidate_multiplier": 2,
        "ann_seed": 7,
    }
    defaults.update(config_overrides)
    return StateBundleInference(
        _DeterministicModel(relevance, aspects),
        StateBundleInferenceConfig(**defaults),
        token_estimator=_UnitTokenEstimator(),
    )


def _all_keys(value: Any) -> set[str]:
    result: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            result.add(str(key).lower())
            result.update(_all_keys(item))
    elif isinstance(value, list):
        for item in value:
            result.update(_all_keys(item))
    return result


def _selected_observations(result: Any) -> tuple[CanonicalObservation, ...]:
    selected: list[CanonicalObservation] = []
    for group in result.groups:
        selected.append(group.anchor.observation)
        selected.extend(group.evidence)
    return tuple(selected)


def test_all_five_channels_parse_as_causally_available_observations() -> None:
    incident = _incident()

    assert tuple(item.channel.value for item in incident.observations) == CHANNELS
    assert incident.query_time_seconds == 10.0
    assert incident.query_watermark_sequence == "cut-query-1"
    assert incident.observations[0].metadata.entities[0].role.value == "affected"
    for observation in incident.observations:
        observation.validate_available(
            incident.query_time_seconds,
            incident.query_watermark_sequence,
        )


def test_live_snapshot_accepts_timestamp_only_causal_availability() -> None:
    snapshot = _snapshot()
    snapshot.pop("query_watermark_sequence")
    for observation in snapshot["observations"]:
        observation["metadata"]["available_at_sequence"] = None

    incident = parse_canonical_snapshot(snapshot)

    assert incident.query_watermark_sequence is None
    assert len(incident.observations) == len(CHANNELS)


@pytest.mark.parametrize(
    "failure", ["future_event", "future_availability", "watermark"]
)
def test_canonical_parser_rejects_future_or_wrong_watermark_observations(
    failure: str,
) -> None:
    snapshot = _snapshot()
    poisoned = deepcopy(snapshot["observations"][0])
    if failure == "future_event":
        poisoned["window"]["end_time_seconds"] = 11.0
        poisoned["metadata"]["event_end_time_seconds"] = 11.0
        poisoned["metadata"]["ingest_time_seconds"] = 11.0
        poisoned["metadata"]["available_at_time_seconds"] = 11.0
    elif failure == "future_availability":
        poisoned["metadata"]["ingest_time_seconds"] = 11.0
        poisoned["metadata"]["available_at_time_seconds"] = 11.0
    else:
        poisoned["metadata"]["available_at_sequence"] = "cut-future"
    snapshot["observations"][0] = poisoned

    with pytest.raises(ValueError):
        CanonicalIncident.from_dict(snapshot)


@pytest.mark.parametrize(
    "forbidden_key",
    (
        "evaluator",
        "expected",
        "active_fault",
        "ground-truth",
        "root cause",
        "success.criteria",
    ),
)
def test_observable_boundary_rejects_evaluator_keys_and_punctuation_aliases(
    forbidden_key: str,
) -> None:
    observation = _observation_dict("log", 0)
    observation["payload"][forbidden_key] = "oracle-secret"

    with pytest.raises(ValueError, match="training-only field"):
        CanonicalObservation.from_dict(
            observation,
            query_time_seconds=10.0,
            query_watermark_sequence="cut-query-1",
        )


def test_runtime_action_context_keeps_controls_but_strips_response_metadata() -> None:
    normalized = _normalize_action_context(
        {
            "action_type": "throttle_workload",
            "parameters": {
                "request_rate_per_second": 1_000,
                "action_schema_ref": "/agent/action-space",
                "action_type": "response-copy",
                "episode_id": "episode-secret",
                "sim_time_seconds": 100,
                "sim_time_seconds_before": 99,
                "sim_time_seconds_after": 100,
                "root/cause": "oracle-secret",
                "ground.truth": "oracle-secret",
            },
        }
    )

    assert normalized == {
        "action_type": "throttle_workload",
        "parameters": {"request_rate_per_second": 1_000},
    }


def test_inference_accepts_only_canonical_incident_and_never_reads_sidecar() -> None:
    incident = _incident([_observation_dict("log", 0)])
    inference = _service([0.9], [[1.0, 0.0]], max_anchors=1)

    class PoisonSidecar:
        def __getattribute__(self, name: str) -> Any:
            if name.startswith("__"):
                return object.__getattribute__(self, name)
            raise AssertionError(f"inference read hidden sidecar field {name}")

    with pytest.raises(TypeError, match="CanonicalIncident"):
        inference.infer(PoisonSidecar())  # type: ignore[arg-type]

    result = inference.infer(incident)
    encoded = result.to_json()
    assert "prototype-truth-secret" not in encoded
    assert "oracle-mechanism-secret" not in encoded
    assert "oracle-target-secret" not in encoded
    assert inference.model.seen_batch_types == [IncidentBatch]


def test_agent_output_is_redacted_and_byte_deterministic() -> None:
    incident = _incident()
    count = len(incident.observations)
    inference = _service(
        [0.9 - index * 0.1 for index in range(count)],
        [[1.0, float(index)] for index in range(count)],
        top_m=4,
        total_token_budget=4,
    )

    first = inference.infer(incident)
    second = inference.infer(incident)
    first_bytes = first.to_json().encode("utf-8")
    second_bytes = second.to_json().encode("utf-8")

    assert first_bytes == second_bytes
    visible = first.to_dict()
    visible_keys = _all_keys(visible)
    assert not (visible_keys & FORBIDDEN_AGENT_KEYS)
    encoded = first_bytes.decode("utf-8")
    assert "private-source-" not in encoded
    assert "private-correlation-" not in encoded
    assert "prototype-truth" not in encoded


def test_inference_enforces_top_m_budgets_and_global_duplicate_suppression() -> None:
    duplicate = _observation_dict("log", 0, source_reference="duplicate-source")
    duplicate_copy = deepcopy(duplicate)
    duplicate_copy["observation_id"] = "obs-001"
    observations = [
        duplicate,
        duplicate_copy,
        _observation_dict("metric", 2),
        _observation_dict("alert", 3),
        _observation_dict("trace", 4),
        _observation_dict("config", 5),
    ]
    incident = _incident(observations)
    inference = _service(
        [0.1, 0.95, 0.8, 0.7, 0.6, 0.5],
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
            [-1.0, 0.0],
            [0.0, -1.0],
        ],
        top_m=3,
        anchor_token_budget=2,
        total_token_budget=3,
        max_anchors=2,
        supports_per_anchor=1,
    )

    result = inference.infer(incident)
    selected = _selected_observations(result)
    selected_ids = [item.observation_id for item in selected]
    signatures = [
        (item.channel.value, tuple(item.metadata.source_references))
        for item in selected
    ]

    assert result.candidate_count == 3
    assert len(result.groups) <= 2
    assert result.used_tokens <= result.token_budget == 3
    assert sum(group.anchor.token_cost for group in result.groups) <= 2
    assert "obs-000" not in selected_ids
    assert "obs-001" in selected_ids
    assert len(signatures) == len(set(signatures))
    assert len(selected_ids) == len(set(selected_ids))


def test_projected_ann_query_is_bounded_and_never_materializes_n_by_n() -> None:
    count = 64
    dimension = 8
    generator = torch.Generator().manual_seed(19)
    vectors = torch.randn((count, dimension), generator=generator)
    identifiers = tuple(f"obs-{index:03d}" for index in range(count))
    index = ProjectedANNIndex(
        vectors,
        identifiers,
        projection_count=3,
        candidate_multiplier=3,
        seed=23,
    )

    result = index.query(7, 5, exclude=frozenset({8, 9}))
    repeated = index.query(7, 5, exclude=frozenset({8, 9}))

    assert result == repeated
    assert len(result) == 5
    assert 7 not in result and 8 not in result and 9 not in result
    assert len(set(result)) == len(result)
    tensor_state = [
        value for value in vars(index).values() if isinstance(value, Tensor)
    ]
    tensor_state.extend(index.sorted_values)
    tensor_state.extend(index.sorted_indices)
    assert all(tuple(value.shape) != (count, count) for value in tensor_state)
    assert max(value.numel() for value in tensor_state) < count * count
