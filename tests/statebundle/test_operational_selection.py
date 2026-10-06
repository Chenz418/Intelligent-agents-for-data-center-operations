"""Synthetic operational safeguards for StateBundle inference.

The fixtures in this module are deliberately small, canonical-shaped, and
model-independent.  An ID-keyed stub supplies adversarial learned relevance
scores so the tests exercise observable inference policy rather than a trained
checkpoint or an LLM.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import statistics
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import pytest
import torch
from torch import nn

from aiopslab.agent_telemetry import (
    AgentObservationRequest,
    compact_statebundle_bundle_token_cost,
    render_snapshot,
)
from aiopslab.statebundle.config import StateBundleInferenceConfig
from aiopslab.statebundle.inference import (
    CharacterTokenEstimator,
    CompactAgentTokenEstimator,
    StateBundleInference,
)
from aiopslab.statebundle.types import (
    CANONICAL_SCHEMA_VERSION,
    CanonicalIncident,
    CanonicalObservation,
    IncidentBatch,
    RedactionPolicy,
)


QUERY_TIME = 100.0
DEFAULT_CUT = "cut-operational-001"
FORBIDDEN_VISIBLE_KEYS = {
    "active_fault",
    "active_faults",
    "annotations",
    "causal_role",
    "evaluator",
    "expected",
    "fault_mechanism",
    "fault_target",
    "fault_type",
    "ground_truth",
    "hidden_training_annotations",
    "inference_visible_target",
    "root_cause",
    "score_hints",
    "success_criteria",
    "training_annotations",
    "training_labels",
}


def _quality() -> dict[str, Any]:
    return {
        "parse_confidence": 1.0,
        "missingness_fraction": 0.0,
        "delay_seconds": 0.0,
        "availability_mask": {"payload": True, "window": True},
        "validation_flags": [],
    }


def _observation(
    channel: str,
    observation_id: str,
    payload: dict[str, Any],
    *,
    start: float = 96.0,
    end: float = QUERY_TIME,
    subsystem: str = "operations",
    entity_id: str = "datacenter",
    entity_role: str = "producer",
    correlation: str | None = None,
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
            "available_at_sequence": DEFAULT_CUT,
            "entities": [
                {
                    "entity_id": entity_id,
                    "role": entity_role,
                    "confidence": 1.0,
                    "provenance": "canonical_fixture",
                }
            ],
            "primary_subsystem": subsystem,
            "primary_subsystem_provenance": "canonical_adapter",
            "correlation_ids": {"signal_id": correlation or f"signal-{observation_id}"},
            "source_references": [f"canonical://{observation_id}"],
            "data_quality": _quality(),
        },
    }


def _metric(
    observation_id: str,
    metric_name: str,
    entity_id: str,
    values: Sequence[float],
    *,
    baseline: float | None = None,
    baseline_iqr: float = 0.0,
    start: float = 96.0,
    end: float = QUERY_TIME,
    subsystem: str = "operations",
    unit: str = "count",
    correlation: str | None = None,
) -> dict[str, Any]:
    numeric = [float(value) for value in values]
    if len(numeric) < 2:
        raise ValueError("metric fixtures require at least two samples")
    step = (end - start) / (len(numeric) - 1)
    timestamps = [start + index * step for index in range(len(numeric))]
    mean = statistics.fmean(numeric)
    median = statistics.median(numeric)
    reference = numeric[0] if baseline is None else float(baseline)
    payload = {
        "unit_type": "series_segment",
        "metric_name": metric_name,
        "unit": unit,
        "scale": "linear",
        "sample_period_seconds": step,
        "timestamps_seconds": timestamps,
        "values": numeric,
        "missingness_mask": [False] * len(numeric),
        "statistics": {
            "count": len(numeric),
            "last": numeric[-1],
            "max": max(numeric),
            "mean": mean,
            "median": median,
            "min": min(numeric),
            "p95": max(numeric),
            "slope_per_second": ((numeric[-1] - numeric[0]) / max(end - start, 1e-9)),
            "stddev": statistics.pstdev(numeric),
        },
        "normalization_reference": {
            "method": "causal_pre_window_robust",
            "sample_count": 8,
            "median": reference,
            "iqr": float(baseline_iqr),
            "last": reference,
            "reference_end_time_seconds": max(0.0, start - 1.0),
        },
        "resource": {"entity_id": entity_id},
    }
    return _observation(
        "metric",
        observation_id,
        payload,
        start=start,
        end=end,
        subsystem=subsystem,
        entity_id=entity_id,
        entity_role="producer",
        correlation=correlation,
    )


def _alert(
    observation_id: str,
    alert_type: str,
    target: str,
    *,
    severity: str = "critical",
    status: str = "firing",
    start: float = 96.0,
    end: float = QUERY_TIME,
    subsystem: str = "operations",
    fingerprint: str | None = None,
    details: Mapping[str, Any] | None = None,
    threshold: Mapping[str, Any] | None = None,
    correlation: str | None = None,
) -> dict[str, Any]:
    fingerprint = fingerprint or f"{alert_type.lower()}:{target}"
    return _observation(
        "alert",
        observation_id,
        {
            "unit_type": "deduplicated_episode",
            "alert_fingerprint": fingerprint,
            "alert_type": alert_type,
            "message": f"{alert_type} observed on {target}",
            "target": target,
            "status": status,
            "severity": severity,
            "threshold": dict(threshold) if threshold is not None else None,
            "duration_seconds": end - start,
            "details": dict(details or {}),
        },
        start=start,
        end=end,
        subsystem=subsystem,
        entity_id=target,
        entity_role="target",
        correlation=correlation,
    )


def _config(
    observation_id: str,
    path: str,
    value: Any,
    *,
    previous_value: Any = None,
    operation: str = "state",
    change_time: float = 0.0,
    scope: str = "datacenter",
    subsystem: str = "configuration",
) -> dict[str, Any]:
    if isinstance(value, bool):
        value_type = "boolean"
    elif isinstance(value, int):
        value_type = "integer"
    elif isinstance(value, float):
        value_type = "number"
    elif value is None:
        value_type = "null"
    else:
        value_type = "string"
    return _observation(
        "config",
        observation_id,
        {
            "unit_type": "scoped_path_value",
            "scope": scope,
            "path": path,
            "operation": operation,
            "value_type": value_type,
            "previous_value": previous_value,
            "value": value,
            "change_time_seconds": change_time,
        },
        start=change_time,
        end=change_time,
        subsystem=subsystem,
        entity_id=scope,
        entity_role="scope",
    )


def _log(
    observation_id: str,
    *,
    event_type: str = "controller_warning",
    subsystem: str = "operations",
    entity_id: str = "datacenter",
    start: float = 96.0,
    end: float = QUERY_TIME,
) -> dict[str, Any]:
    return _observation(
        "log",
        observation_id,
        {
            "unit_type": "template_aggregate",
            "template_id": f"template-{observation_id}",
            "template": f"{event_type} reported by {{entity}}",
            "event_type": event_type,
            "count": 4,
            "severity": "warning",
            "severity_histogram": {"warning": 4},
            "rarity": 0.25,
            "burst_rate_per_minute": 2.0,
            "variable_summaries": {"entity": {"top": entity_id}},
            "time_features": {
                "first_offset_seconds": 0.0,
                "last_offset_seconds": end - start,
            },
        },
        start=start,
        end=end,
        subsystem=subsystem,
        entity_id=entity_id,
    )


def _trace(
    observation_id: str,
    *,
    source: str = "tenant:default",
    destination: str = "service:web",
    start: float = 96.0,
    end: float = QUERY_TIME,
) -> dict[str, Any]:
    return _observation(
        "trace",
        observation_id,
        {
            "unit_type": "source_destination_operation_group",
            "operation": "serve_request",
            "source": source,
            "destination": destination,
            "count": 10.0,
            "status_counts": {"ok": 9.0, "error": 1.0},
            "retry_count": 1.0,
            "latency_ms": {"mean": 12.0, "p95": 20.0, "min": 8.0, "max": 20.0},
            "critical_path": True,
        },
        start=start,
        end=end,
        subsystem="application",
        entity_id=source,
        entity_role="source",
    )


def _incident(
    observations: Sequence[dict[str, Any]],
    *,
    query_time: float = QUERY_TIME,
    cut_index: int = 1,
    incident_id: str = "episode-operational-selection",
) -> CanonicalIncident:
    watermark = f"cut-operational-{cut_index:03d}"
    copied = deepcopy(list(observations))
    for item in copied:
        item["metadata"]["available_at_sequence"] = watermark
    return CanonicalIncident.from_dict(
        {
            "schema_version": CANONICAL_SCHEMA_VERSION,
            "episode_id": incident_id,
            "snapshot_id": f"snapshot-operational-{cut_index:03d}",
            "query_time_seconds": query_time,
            "query_watermark_sequence": watermark,
            "observations": copied,
        }
    )


class _ScoresByIdModel(nn.Module):
    """Observable-only stub whose output is stable under repeated calls."""

    def __init__(self, scores: Mapping[str, float], *, default: float = 0.1):
        super().__init__()
        self.scores = dict(scores)
        self.default = float(default)
        self.seen_batch_types: list[type[Any]] = []

    @staticmethod
    def _aspect(observation_id: str) -> list[float]:
        digest = hashlib.sha256(observation_id.encode("utf-8")).digest()
        return [0.05 + digest[index] / 255.0 for index in range(4)]

    def forward(self, batch: IncidentBatch) -> SimpleNamespace:
        if not isinstance(batch, IncidentBatch):
            raise AssertionError("model received a non-observable batch")
        self.seen_batch_types.append(type(batch))
        observations = batch.incidents[0].observations
        relevance = torch.tensor(
            [
                self.scores.get(item.observation_id, self.default)
                for item in observations
            ],
            dtype=torch.float32,
        )
        aspects = torch.tensor(
            [self._aspect(item.observation_id) for item in observations],
            dtype=torch.float32,
        )
        return SimpleNamespace(
            relevance_scores=relevance.unsqueeze(0),
            aspect_embeddings=aspects.unsqueeze(0),
        )


class _CostsById:
    def __init__(self, costs: Mapping[str, int] | None = None, *, default: int = 1):
        self.costs = dict(costs or {})
        self.default = default

    def estimate(self, observation: CanonicalObservation, policy: Any) -> int:
        del policy
        return self.costs.get(observation.observation_id, self.default)

    def estimate_bundle(
        self,
        groups: Sequence[tuple[CanonicalObservation, Sequence[CanonicalObservation]]],
        policy: Any,
        *,
        query_time_seconds: float,
        request: AgentObservationRequest | None,
    ) -> int:
        del policy, query_time_seconds, request
        seen: set[str] = set()
        total = 0
        for anchor, supports in groups:
            for observation in (anchor, *supports):
                if observation.observation_id in seen:
                    continue
                seen.add(observation.observation_id)
                total += self.costs.get(observation.observation_id, self.default)
        return total


class _RowsPlusGroupCount(_CostsById):
    """Expose the difference between a deferred anchor and fitting support."""

    def estimate_bundle(
        self,
        groups: Sequence[tuple[CanonicalObservation, Sequence[CanonicalObservation]]],
        policy: Any,
        *,
        query_time_seconds: float,
        request: AgentObservationRequest | None,
    ) -> int:
        row_cost = super().estimate_bundle(
            groups,
            policy,
            query_time_seconds=query_time_seconds,
            request=request,
        )
        return row_cost + len(groups)


def _service(
    scores: Mapping[str, float],
    *,
    budget: int = 4,
    costs: Mapping[str, int] | None = None,
    **overrides: Any,
) -> StateBundleInference:
    defaults: dict[str, Any] = {
        "top_m": max(1, len(scores)),
        "anchor_token_budget": budget,
        "total_token_budget": budget,
        "max_anchors": 4,
        "supports_per_anchor": 2,
        "diversity_weight": 0.0,
        "minimum_anchor_gain": 0.0,
        "minimum_support_score": -10.0,
        "minimum_budget_fill_score": -10.0,
        "ann_projection_count": 2,
        "ann_candidate_multiplier": 4,
        "ann_seed": 17,
    }
    defaults.update(overrides)
    return StateBundleInference(
        _ScoresByIdModel(scores),
        StateBundleInferenceConfig(**defaults),
        token_estimator=_CostsById(costs),
    )


def _selected(result: Any) -> tuple[CanonicalObservation, ...]:
    return tuple(
        item
        for group in result.groups
        for item in (group.anchor.observation, *group.evidence)
    )


def _selected_ids(result: Any) -> set[str]:
    return {item.observation_id for item in _selected(result)}


def _audit_candidate(result: Any, observation_id: str) -> dict[str, Any]:
    return next(
        item
        for item in result.audit_dict()["candidates"]
        if item["observation_id"] == observation_id
    )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_character_token_estimator_rejects_non_finite_rates(value: float) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        CharacterTokenEstimator(characters_per_token=value)

    with pytest.raises(ValueError, match="finite and positive"):
        StateBundleInferenceConfig(characters_per_token=value)


def _all_keys(value: Any) -> set[str]:
    if isinstance(value, Mapping):
        return {str(key).lower() for key in value} | {
            key for item in value.values() for key in _all_keys(item)
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return {key for item in value for key in _all_keys(item)}
    return set()


def _assert_budget(result: Any) -> None:
    assert result.used_tokens <= result.token_budget
    audit = result.audit_dict()
    assert audit["budget_accounted_telemetry_tokens"] == result.used_tokens
    assert audit["token_accounting_exact"] is False
    assert audit["token_budget_semantics"] == "custom_estimator_units"
    assert audit["actual_serialized_telemetry_tokens"] >= 0
    assert audit["unused_tokens"] == result.token_budget - result.used_tokens


def test_critical_alert_beats_static_config() -> None:
    alert = _alert("alert-critical", "SchedulerAPILatencyHigh", "scheduler")
    static = _config(
        "config-static",
        "simulation.auto_advance",
        False,
        operation="set",
        change_time=0.0,
    )
    service = _service(
        {"alert-critical": 0.01, "config-static": 0.99},
        budget=1,
        top_m=1,
    )

    result = service.infer(_incident([static, alert]))

    assert _selected_ids(result) == {"alert-critical"}
    alert_audit = _audit_candidate(result, "alert-critical")
    assert alert_audit["observable_features"]["active_alert"] is True
    assert alert_audit["selection_phase"] == "protected"
    assert result.audit_dict()["learned_top_m_coverage_misses"]
    _assert_budget(result)


def test_protected_support_placement_ignores_ordinary_score_thresholds() -> None:
    first = _alert("protected-a", "AlertA", "rack-r1-row1-01")
    second = _alert("protected-b", "AlertB", "rack-r1-row1-02")
    config = StateBundleInferenceConfig(
        top_m=2,
        anchor_token_budget=3,
        total_token_budget=3,
        max_anchors=2,
        supports_per_anchor=1,
        minimum_support_score=1_000.0,
        minimum_budget_fill_score=1_000.0,
        ann_projection_count=2,
        ann_candidate_multiplier=2,
    )
    service = StateBundleInference(
        _ScoresByIdModel({"protected-a": 0.5, "protected-b": 0.4}),
        config,
        token_estimator=_RowsPlusGroupCount(),
    )

    result = service.infer(_incident([first, second]))

    assert _selected_ids(result) == {"protected-a", "protected-b"}
    assert len(result.groups) == 1
    assert result.audit_dict()["protected_omissions"] == []
    assert [
        item["observation_id"]
        for item in result.audit_dict()["protected_anchor_deferrals"]
    ] == ["protected-b"]


def test_protected_rows_use_cheapest_shared_envelope_when_all_fit() -> None:
    alert_dicts = [
        _alert(
            f"protected-repack-{index}",
            "RackTrafficDrop",
            f"rack-r1-row1-{index + 1:02d}",
        )
        for index in range(3)
    ]
    incident = _incident(alert_dicts)
    observations = [item.to_agent_dict() for item in incident.observations]
    references = [
        {
            "group": 0,
            "anchor": "protected-repack-0",
            "supports": ["protected-repack-1", "protected-repack-2"],
        }
    ]
    target_scope_candidates = [
        {
            "scope": f"rack-r1-row1-{index + 1:02d}",
            "estimated_role": "direct_target_candidate",
            "supporting_observation_ids": [f"protected-repack-{index}"],
        }
        for index in range(3)
    ]
    exact_budget = compact_statebundle_bundle_token_cost(
        observations,
        references,
        query_time_seconds=QUERY_TIME,
        target_scope_ambiguity=True,
        target_scope_candidates=target_scope_candidates,
    )
    service = StateBundleInference(
        _ScoresByIdModel(
            {
                "protected-repack-0": 0.3,
                "protected-repack-1": 0.2,
                "protected-repack-2": 0.1,
            }
        ),
        StateBundleInferenceConfig(
            top_m=3,
            anchor_token_budget=exact_budget,
            total_token_budget=exact_budget,
        ),
    )

    result = service.infer(incident)

    assert _selected_ids(result) == {
        "protected-repack-0",
        "protected-repack-1",
        "protected-repack-2",
    }
    assert len(result.groups) == 1
    assert result.used_tokens == exact_budget
    assert result.audit_dict()["protected_omissions"] == []
    assert result.audit_dict()["protected_group_repacks"] == []
    deferrals = result.audit_dict()["protected_anchor_deferrals"]
    assert [item["observation_id"] for item in deferrals] == [
        "protected-repack-1",
        "protected-repack-2",
    ]
    assert {item["reason"] for item in deferrals} == {
        "protected_anchor_higher_token_cost",
        "protected_anchor_envelope_overflow",
    }
    rendered = render_snapshot(result.to_dict(), condition="statebundle")
    assert rendered["budget"]["serialization_admission_dropped_count"] == 0
    assert rendered["budget"]["actual_serialized_tokens"] == result.used_tokens
    assert rendered["budget"]["actual_serialized_tokens"] <= exact_budget


def test_protected_redundant_rows_are_capped_before_shared_group_budget_fill() -> None:
    alerts = [
        _alert(
            f"protected-redundant-{index}",
            "RackTrafficDrop",
            "rack-r1-row1-01",
            details={"distinct_member": index},
        )
        for index in range(4)
    ]
    config = StateBundleInferenceConfig(
        top_m=4,
        anchor_token_budget=7,
        total_token_budget=7,
        max_anchors=1,
        supports_per_anchor=1,
        minimum_support_score=100.0,
        minimum_budget_fill_score=-10.0,
        ann_projection_count=2,
        ann_candidate_multiplier=2,
        semantic_group_representatives=2,
    )
    service = StateBundleInference(
        _ScoresByIdModel(
            {f"protected-redundant-{index}": 1.0 - index / 10 for index in range(4)}
        ),
        config,
        token_estimator=_RowsPlusGroupCount(),
    )

    result = service.infer(_incident(alerts))

    assert _selected_ids(result) == {
        "protected-redundant-0",
        "protected-redundant-1",
    }
    assert result.used_tokens == 3
    assert result.token_budget == 7
    assert result.audit_dict()["unused_tokens"] == 4
    assert result.audit_dict()["unused_capacity_reason"] == "no_candidate_fits"
    omissions = result.audit_dict()["protected_omissions"]
    assert {item["observation_id"] for item in omissions} == {
        "protected-redundant-2",
        "protected-redundant-3",
    }
    assert {item["reason"] for item in omissions} == {"protected_equivalence_cap"}
    for observation_id in ("protected-redundant-2", "protected-redundant-3"):
        candidate = _audit_candidate(result, observation_id)
        assert candidate["semantic_redundant"] is True
        assert candidate["selected_as"] == "rejected"
        assert candidate["selection_phase"] is None
        assert candidate["rejection_reason"] == "protected_equivalence_cap"
        assert candidate["equivalence_representative_id"] == ("protected-redundant-0")


def test_target_alert_survives_localization_without_oracle_target() -> None:
    target_alert = _alert(
        "alert-target",
        "RackTrafficDrop",
        "rack-r1-row1-01",
        severity="warning",
        subsystem="network",
    )
    distractors = [
        _config(
            f"config-{index}",
            f"topology.value_{index}",
            index,
            operation="set",
            change_time=0.0,
        )
        for index in range(3)
    ]
    scores = {"alert-target": 0.01, **{f"config-{index}": 0.99 for index in range(3)}}
    service = _service(scores, budget=1, top_m=1)

    result = service.infer(_incident([*distractors, target_alert]))

    assert _selected_ids(result) == {"alert-target"}
    candidate = _audit_candidate(result, "alert-target")
    assert candidate["observable_features"]["target_bearing"] is True
    assert "active_target_alert" in candidate["protection_reasons"]
    assert not (
        {"fault_target", "expected", "root_cause"} & _all_keys(result.to_dict())
    )
    _assert_budget(result)


def test_flat_zero_fanout_is_capped_and_normal_comparison_is_retained() -> None:
    alert = _alert(
        "alert-cpu",
        "RackTrafficDrop",
        "rack-r1-row1-01",
        severity="warning",
        subsystem="network",
        details={"metric": "rack.cpu_utilization"},
    )
    target = _metric(
        "metric-cpu-target",
        "rack.cpu_utilization",
        "rack-r1-row1-01",
        [8.33, 8.33, 8.33, 0.0],
        baseline=8.33,
        subsystem="network",
        unit="percent",
    )
    peer = _metric(
        "metric-cpu-peer",
        "rack.cpu_utilization",
        "rack-r1-row1-02",
        [10.0, 10.0, 10.0, 10.0],
        baseline=10.0,
        subsystem="network",
        unit="percent",
    )
    zeros = [
        _metric(
            f"metric-zero-{index}",
            "rack.failed_server_count",
            f"rack-r1-row2-{index + 1:02d}",
            [0.0, 0.0, 0.0, 0.0],
            baseline=0.0,
            subsystem="compute",
        )
        for index in range(6)
    ]
    scores = {
        "alert-cpu": 0.01,
        "metric-cpu-target": 0.05,
        "metric-cpu-peer": 0.50,
        **{f"metric-zero-{index}": 0.99 for index in range(6)},
    }
    service = _service(scores, budget=5)

    result = service.infer(_incident([*zeros, peer, target, alert]))
    selected = _selected_ids(result)

    assert {"alert-cpu", "metric-cpu-target", "metric-cpu-peer"} <= selected
    selected_zeros = selected & {f"metric-zero-{index}" for index in range(6)}
    assert len(selected_zeros) <= service.config.zero_series_representatives
    assert (
        _audit_candidate(result, "metric-cpu-peer")["observable_features"][
            "normal_comparison"
        ]
        is True
    )
    assert (
        _audit_candidate(result, "metric-cpu-target")["observable_features"][
            "zero_series"
        ]
        is False
    )
    assert any(
        _audit_candidate(result, f"metric-zero-{index}")["semantic_redundant"]
        for index in range(6)
    )
    _assert_budget(result)


def test_content_duplicate_fallback_ignores_ids_without_collapsing_correlations() -> (
    None
):
    duplicate_a = _log("content-duplicate-a", event_type="controller_warning")
    duplicate_b = deepcopy(duplicate_a)
    duplicate_b["observation_id"] = "content-duplicate-b"
    for duplicate in (duplicate_a, duplicate_b):
        duplicate["metadata"]["source_references"] = []
        duplicate["metadata"]["correlation_ids"] = {}

    duplicate_result = _service(
        {"content-duplicate-a": 0.1, "content-duplicate-b": 0.9}, budget=2
    ).infer(
        _incident([duplicate_a, duplicate_b]),
        request=AgentObservationRequest(channels=("log",), detail="raw"),
    )

    assert _selected_ids(duplicate_result) == {"content-duplicate-b"}
    assert (
        _audit_candidate(duplicate_result, "content-duplicate-a")["rejection_reason"]
        == "exact_duplicate"
    )
    assert duplicate_result.audit_dict()["unique_candidate_count"] == 1
    request_status = duplicate_result.to_dict()["request_status"]
    assert request_status["matched_observation_count"] == 2
    assert request_status["unique_matched_fact_count"] == 1
    assert request_status["duplicate_matched_observation_count"] == 1
    assert request_status["selected_match_count"] == 1

    metric_a = _metric(
        "shared-correlation-cpu",
        "rack.cpu_utilization",
        "server-1",
        [10.0, 80.0],
        baseline=10.0,
        correlation="shared-episode",
    )
    metric_b = _metric(
        "shared-correlation-network",
        "network.latency",
        "server-2",
        [5.0, 50.0],
        baseline=5.0,
        correlation="shared-episode",
    )
    metric_a["metadata"]["source_references"] = []
    metric_b["metadata"]["source_references"] = []
    correlation_result = _service(
        {"shared-correlation-cpu": 0.9, "shared-correlation-network": 0.8},
        budget=2,
    ).infer(_incident([metric_a, metric_b]))

    assert _selected_ids(correlation_result) == {
        "shared-correlation-cpu",
        "shared-correlation-network",
    }
    assert correlation_result.audit_dict()["unique_candidate_count"] == 2


def test_recent_changed_config_is_selectable_and_static_config_is_downranked() -> None:
    changed = _config(
        "config-changed",
        "controls.scheduler.retry_limit",
        4,
        previous_value=2,
        operation="update",
        change_time=99.0,
        scope="scheduler",
        subsystem="control_plane",
    )
    static = _config(
        "config-bootstrap",
        "simulation.auto_advance",
        False,
        previous_value=None,
        operation="set",
        change_time=0.0,
    )
    unchanged_update = _config(
        "config-no-op-update",
        "controls.scheduler.batch_size",
        16,
        previous_value=16,
        operation="update",
        change_time=99.0,
        scope="scheduler",
        subsystem="control_plane",
    )
    service = _service(
        {
            "config-changed": 0.40,
            "config-bootstrap": 0.70,
            "config-no-op-update": 0.70,
        },
        budget=1,
    )

    result = service.infer(_incident([static, unchanged_update, changed]))

    assert _selected_ids(result) == {"config-changed"}
    changed_audit = _audit_candidate(result, "config-changed")
    static_audit = _audit_candidate(result, "config-bootstrap")
    unchanged_audit = _audit_candidate(result, "config-no-op-update")
    assert changed_audit["observable_features"]["recent_config_change"] is True
    assert changed_audit["reranking_contributions"]["recent_config_change"] > 0
    assert static_audit["observable_features"]["static_config"] is True
    assert static_audit["reranking_contributions"]["static_config"] < 0
    assert unchanged_audit["observable_features"]["recent_config_change"] is False
    assert unchanged_audit["observable_features"]["static_config"] is True
    assert unchanged_audit["reranking_contributions"]["static_config"] < 0
    _assert_budget(result)


def test_targeted_metric_entity_request_changes_selection_before_rendering() -> None:
    target = _metric(
        "metric-requested",
        "control_plane.scheduler_api_latency",
        "scheduler",
        [75.0, 75.0, 75.0, 75.0],
        baseline=75.0,
        subsystem="control_plane",
        unit="milliseconds",
    )
    distractor = _log("log-unrelated", event_type="rare_controller_failure")
    incident = _incident([target, distractor])
    scores = {"metric-requested": 0.01, "log-unrelated": 0.99}

    ordinary = _service(scores, budget=1).infer(incident)
    targeted = _service(scores, budget=1).infer(
        incident,
        request=AgentObservationRequest(
            channels=("metric",),
            detail="raw",
            metric_names=("control_plane.scheduler_api_latency",),
            entity_ids=("scheduler",),
        ),
    )

    assert _selected_ids(ordinary) == {"log-unrelated"}
    assert _selected_ids(targeted) == {"metric-requested"}
    assert targeted.to_dict()["request_status"]["status"] == "satisfied"
    assert targeted.to_dict()["request_status"]["selected_match_count"] == 1
    assert (
        _audit_candidate(targeted, "metric-requested")["observable_features"][
            "request_match"
        ]
        is True
    )
    _assert_budget(targeted)


def test_explicit_all_channels_is_cache_equivalent_to_default_request() -> None:
    older = _log("all-channels-older", event_type="diagnostic_notice", end=99.0)
    newer = _log("all-channels-newer", event_type="diagnostic_notice", end=100.0)
    incident = _incident([newer, older])
    scores = {"all-channels-older": 1.0, "all-channels-newer": 0.0}
    explicit_all = AgentObservationRequest(
        channels=("log", "metric", "alert", "trace", "config")
    )

    default_first_service = _service(scores, budget=1)
    default_first = default_first_service.infer(incident)
    explicit_second = default_first_service.infer(incident, request=explicit_all)

    explicit_first_service = _service(scores, budget=1)
    explicit_first = explicit_first_service.infer(incident, request=explicit_all)
    default_second = explicit_first_service.infer(incident)

    assert _selected_ids(default_first) == {"all-channels-older"}
    assert (
        default_first.to_json()
        == explicit_second.to_json()
        == explicit_first.to_json()
        == default_second.to_json()
    )
    assert all(
        result.audit_dict()["targeted_request"] is False
        for result in (default_first, explicit_second, explicit_first, default_second)
    )


def test_targeted_request_scope_uses_agent_visible_redacted_representation() -> None:
    metric = _metric(
        "redacted-request-metric",
        "cpu.util",
        "server-1",
        [10.0, 80.0],
        baseline=10.0,
    )
    request = AgentObservationRequest(
        channels=("metric",),
        detail="raw",
        metric_names=("cpu.util",),
    )
    config = StateBundleInferenceConfig(
        top_m=1,
        redaction_policy=RedactionPolicy(
            include_payload=False,
            include_entities=False,
            include_primary_subsystem=False,
        ),
    )
    result = StateBundleInference(
        _ScoresByIdModel({"redacted-request-metric": 1.0}), config
    ).infer(_incident([metric]), request=request)

    assert _selected_ids(result) == set()
    assert result.to_dict()["request_status"]["status"] == (
        "no_matching_canonical_observation"
    )
    assert (
        _audit_candidate(result, "redacted-request-metric")["rejection_reason"]
        == "outside_request_scope"
    )
    rendered = render_snapshot(
        result.to_dict(),
        condition="statebundle",
        request=request,
    )
    assert rendered["tables"] == {}
    assert rendered["evidence_groups"] == []


def test_exact_deduplication_uses_the_configured_visible_projection() -> None:
    first = _log("redacted-duplicate-a", event_type="first_private_payload")
    second = _log("redacted-duplicate-b", event_type="second_private_payload")
    scores = {"redacted-duplicate-a": 0.1, "redacted-duplicate-b": 0.9}
    hidden_payload_policy = RedactionPolicy(include_payload=False)
    collapsed = _service(
        scores,
        budget=2,
        redaction_policy=hidden_payload_policy,
    ).infer(_incident([first, second]))

    assert _selected_ids(collapsed) == {"redacted-duplicate-b"}
    assert collapsed.audit_dict()["unique_candidate_count"] == 1
    assert (
        _audit_candidate(collapsed, "redacted-duplicate-a")["rejection_reason"]
        == "exact_duplicate"
    )

    correlations_visible = _service(
        scores,
        budget=2,
        redaction_policy=RedactionPolicy(
            include_payload=False,
            include_correlation_ids=True,
        ),
    ).infer(_incident([first, second]))
    assert _selected_ids(correlations_visible) == {
        "redacted-duplicate-a",
        "redacted-duplicate-b",
    }
    assert correlations_visible.audit_dict()["unique_candidate_count"] == 2


def test_request_scope_excludes_config_and_applies_zero_log_limit() -> None:
    static = _config(
        "request-static",
        "simulation.auto_advance",
        False,
        operation="state",
        change_time=0.0,
    )
    log = _log("request-log", event_type="rare_controller_failure")
    metric = _metric(
        "request-metric",
        "workload.queue_length",
        "workload",
        [1.0, 1.0],
        baseline=1.0,
    )
    scores = {"request-static": 1.0, "request-log": 0.8, "request-metric": 0.1}
    incident = _incident([static, log, metric])

    without_config = _service(scores, budget=1).infer(
        incident,
        request=AgentObservationRequest(include_config=False),
    )
    without_logs = _service(scores, budget=1).infer(
        incident,
        request=AgentObservationRequest(include_config=False, log_limit=0),
    )

    assert _selected_ids(without_config) == {"request-log"}
    assert _audit_candidate(without_config, "request-static")["rejection_reason"] == (
        "outside_request_scope"
    )
    assert _selected_ids(without_logs) == {"request-metric"}
    assert _audit_candidate(without_logs, "request-log")["rejection_reason"] == (
        "outside_request_scope"
    )
    assert without_logs.to_dict()["request_status"]["status"] == "satisfied"
    assert without_logs.to_dict()["request_status"]["selected_match_count"] == 1


def test_log_limit_is_applied_jointly_to_current_and_retained_rows() -> None:
    old = _log(
        "retained-old-log",
        event_type="request_queue_warning_old",
        end=100.0,
    )
    new = _log(
        "current-new-log",
        event_type="request_queue_warning_new",
        end=101.0,
    )
    service = _service({"retained-old-log": 0.9, "current-new-log": 0.8}, budget=2)
    service.infer(_incident([old], cut_index=1))

    result = service.infer(
        _incident([new], query_time=101.0, cut_index=2),
        request=AgentObservationRequest(channels=("log",), log_limit=1),
    )

    assert _selected_ids(result) == {"current-new-log"}
    old_audit = _audit_candidate(result, "retained-old-log")
    assert old_audit["rejection_reason"] == "outside_request_scope"
    assert old_audit["observable_features"]["request_eligible"] is False
    assert result.to_dict()["request_status"]["matched_observation_count"] == 1
    assert result.to_dict()["request_status"]["selected_match_count"] == 1


def test_log_limit_can_displace_older_current_row_with_newer_retained_row() -> None:
    retained = _log(
        "retained-newer-log",
        event_type="request_queue_warning_retained",
        end=100.0,
    )
    current = _log(
        "current-older-log",
        event_type="request_queue_warning_current",
        end=99.0,
    )
    service = _service({"retained-newer-log": 0.9, "current-older-log": 0.8}, budget=2)
    service.infer(_incident([retained], cut_index=1))

    result = service.infer(
        _incident([current], query_time=101.0, cut_index=2),
        request=AgentObservationRequest(channels=("log",), log_limit=1),
    )

    assert _selected_ids(result) == {"retained-newer-log"}
    current_audit = _audit_candidate(result, "current-older-log")
    assert current_audit["rejection_reason"] == "outside_request_scope"
    assert current_audit["observable_features"]["request_eligible"] is False
    assert result.to_dict()["request_status"]["matched_observation_count"] == 1
    assert result.to_dict()["request_status"]["selected_match_count"] == 1


def test_joint_request_scope_is_finalized_before_exact_deduplication() -> None:
    retained = _log(
        "joint-scope-retained",
        event_type="request_queue_warning_retained",
        end=102.0,
    )
    current_high = _log(
        "joint-scope-current-high",
        event_type="request_queue_warning_high",
        end=100.0,
    )
    current_low = _log(
        "joint-scope-current-low",
        event_type="request_queue_warning_low",
        end=101.0,
    )
    current_high["metadata"]["source_references"] = ["canonical://shared-current"]
    current_low["metadata"]["source_references"] = ["canonical://shared-current"]
    service = _service(
        {
            "joint-scope-retained": 0.8,
            "joint-scope-current-high": 0.99,
            "joint-scope-current-low": 0.01,
        },
        budget=2,
    )
    service.infer(_incident([retained], query_time=102.0, cut_index=1))

    result = service.infer(
        _incident([current_high, current_low], query_time=103.0, cut_index=2),
        request=AgentObservationRequest(channels=("log",), log_limit=2),
    )

    assert _selected_ids(result) == {
        "joint-scope-retained",
        "joint-scope-current-low",
    }
    high_audit = _audit_candidate(result, "joint-scope-current-high")
    low_audit = _audit_candidate(result, "joint-scope-current-low")
    assert high_audit["rejection_reason"] == "outside_request_scope"
    assert low_audit["rejection_reason"] is None
    assert low_audit["observable_features"]["request_eligible"] is True
    assert result.to_dict()["request_status"]["matched_observation_count"] == 2


def test_shared_source_reference_does_not_collapse_distinct_retained_fact() -> None:
    retained = _log(
        "duplicate-retained-log",
        event_type="request_queue_warning_retained",
    )
    current = _log(
        "duplicate-current-log",
        event_type="request_queue_warning_current",
        end=101.0,
    )
    retained["metadata"]["source_references"] = ["canonical://same-log"]
    current["metadata"]["source_references"] = ["canonical://same-log"]
    service = _service(
        {"duplicate-retained-log": 0.9, "duplicate-current-log": 0.1}, budget=2
    )
    service.infer(_incident([retained], cut_index=1))

    result = service.infer(
        _incident([current], query_time=101.0, cut_index=2),
    )

    assert _selected_ids(result) == {
        "duplicate-current-log",
        "duplicate-retained-log",
    }
    retained_audit = _audit_candidate(result, "duplicate-retained-log")
    assert retained_audit["rejection_reason"] is None
    assert retained_audit["selected_as"] in {"anchor", "support"}


def test_shared_source_reference_does_not_supersede_distinct_log_fact() -> None:
    old = _log(
        "duplicate-memory-old",
        event_type="request_queue_warning_old",
        end=100.0,
    )
    new = _log(
        "duplicate-memory-new",
        event_type="request_queue_warning_new",
        end=101.0,
    )
    distractor = _log(
        "duplicate-memory-distractor",
        event_type="unrelated_notice",
        end=102.0,
    )
    old["metadata"]["source_references"] = ["canonical://evolving-log"]
    new["metadata"]["source_references"] = ["canonical://evolving-log"]
    service = _service(
        {
            "duplicate-memory-old": 10.0,
            "duplicate-memory-new": 1.0,
            "duplicate-memory-distractor": 0.0,
        },
        budget=1,
    )
    service.infer(_incident([old], cut_index=1))

    second = service.infer(
        _incident([new], query_time=101.0, cut_index=2),
    )
    assert _selected_ids(second) == {"duplicate-memory-old"}
    assert not second.audit_dict()["causal_cut_transition"]["memory_evictions"]

    third = service.infer(
        _incident([distractor], query_time=102.0, cut_index=3),
    )
    assert _selected_ids(third) == {"duplicate-memory-old"}


def test_shared_source_does_not_transfer_resolution_streak_between_alert_facts() -> (
    None
):
    alert_a = _alert(
        "streak-alert-a",
        "RackWarning",
        "rack-r1-row1-01",
        fingerprint="streak-a",
    )
    alert_b = _alert(
        "streak-alert-b",
        "RackWarning",
        "rack-r1-row1-01",
        fingerprint="streak-b",
        end=102.0,
    )
    alert_c = _alert(
        "streak-alert-c",
        "RackWarning",
        "rack-r1-row1-01",
        fingerprint="streak-a",
        end=103.0,
    )
    alert_c_resolved = _alert(
        "streak-alert-c-resolved",
        "RackWarning",
        "rack-r1-row1-01",
        fingerprint="streak-a",
        status="resolved",
        end=104.0,
    )
    distractor = _log(
        "streak-distractor",
        event_type="unrelated_notice",
        end=101.0,
    )
    alert_a["metadata"]["source_references"] = ["canonical://shared-alert"]
    alert_b["metadata"]["source_references"] = ["canonical://shared-alert"]
    alert_c["metadata"]["source_references"] = ["canonical://fresh-alert"]
    alert_c_resolved["metadata"]["source_references"] = ["canonical://fresh-alert"]
    service = _service(
        {
            "streak-alert-a": 2.0,
            "streak-alert-b": 1.0,
            "streak-alert-c": 1.0,
            "streak-alert-c-resolved": 1.0,
            "streak-distractor": 0.0,
        },
        budget=3,
        resolution_stability_cuts=3,
    )
    service.infer(_incident([alert_a], cut_index=1))
    service.infer(
        _incident([distractor], query_time=101.0, cut_index=2),
    )
    stale_key = next(iter(service._memory.resolution_streaks))
    assert service._memory.resolution_streaks[stale_key] == 1

    service.infer(
        _incident([alert_b], query_time=102.0, cut_index=3),
    )
    assert service._memory.resolution_streaks[stale_key] == 2

    service.infer(
        _incident([alert_c], query_time=103.0, cut_index=4),
    )
    first_resolved = service.infer(
        _incident([alert_c_resolved], query_time=104.0, cut_index=5),
    )
    assert service._memory.resolution_streaks[stale_key] == 1
    assert not any(
        item["observation_id"] == "streak-alert-c"
        and item["reason"] == "condition_resolved_stably"
        for item in first_resolved.audit_dict()["causal_cut_transition"][
            "memory_evictions"
        ]
    )


def test_unknown_alert_target_is_not_treated_as_localization_identity() -> None:
    alert = _alert(
        "unknown-target-alert",
        "GenericWarning",
        "unknown",
        severity="warning",
    )
    changed = _config(
        "known-config-change",
        "controls.scheduler.retry_limit",
        4,
        previous_value=2,
        operation="update",
        change_time=99.0,
        scope="scheduler",
    )
    result = _service(
        {"unknown-target-alert": 0.01, "known-config-change": 0.99},
        budget=1,
    ).infer(_incident([alert, changed]))

    assert _selected_ids(result) == {"unknown-target-alert"}
    candidate = _audit_candidate(result, "unknown-target-alert")
    assert candidate["observable_features"]["target_bearing"] is False
    assert candidate["protected"] is False
    assert candidate["selection_phase"] == "coverage_guard"
    assert "active_target_alert" not in candidate["protection_reasons"]
    assert result.audit_dict()["coverage_evictions"] == [
        {
            "observation_id": "known-config-change",
            "reason": "evicted_for_coverage:active_alert",
        }
    ]


def test_unknown_entity_link_does_not_create_a_localization_identity() -> None:
    alert = _alert(
        "unknown-link-alert",
        "GenericWarning",
        "unknown",
        severity="warning",
        correlation="shared-unknown-correlation",
    )
    metric = _metric(
        "unknown-link-metric",
        "rack.queue_depth",
        "unknown",
        [1.0, 12.0],
        baseline=1.0,
        correlation="shared-unknown-correlation",
    )
    result = _service(
        {"unknown-link-alert": 0.2, "unknown-link-metric": 0.1}, budget=2
    ).infer(_incident([alert, metric]))

    candidate = _audit_candidate(result, "unknown-link-metric")
    assert candidate["observable_features"]["linked_to_active_alert"] is True
    assert candidate["observable_features"]["target_bearing"] is False
    assert candidate["observable_features"]["target_ids"] == []
    assert "entity_identifier_bonus" not in candidate["reranking_contributions"]
    assert "target_role_bonus" not in candidate["reranking_contributions"]


def test_generic_entity_overlap_alone_does_not_link_anomaly_to_alert() -> None:
    alert = _alert(
        "generic-overlap-alert",
        "GenericWarning",
        "unknown",
        severity="warning",
    )
    metric = _metric(
        "generic-overlap-metric",
        "rack.queue_depth",
        "unknown",
        [1.0, 12.0],
        baseline=1.0,
    )
    result = _service(
        {"generic-overlap-alert": 0.2, "generic-overlap-metric": 0.1}, budget=2
    ).infer(_incident([alert, metric]))

    candidate = _audit_candidate(result, "generic-overlap-metric")
    assert candidate["observable_features"]["anomalous_metric"] is True
    assert candidate["observable_features"]["linked_to_active_alert"] is False
    assert "explicit_active_alert_link" not in candidate["protection_reasons"]


def test_correlation_values_do_not_link_across_different_namespaces() -> None:
    alert = _alert(
        "namespace-alert",
        "GenericWarning",
        "rack-r1-row1-01",
        severity="warning",
    )
    metric = _metric(
        "namespace-metric",
        "scheduler.queue_depth",
        "scheduler",
        [1.0, 12.0],
        baseline=1.0,
    )
    alert["metadata"]["correlation_ids"] = {"alert_episode": "opaque-42"}
    metric["metadata"]["correlation_ids"] = {"trace_id": "opaque-42"}
    result = _service(
        {"namespace-alert": 0.2, "namespace-metric": 0.1}, budget=2
    ).infer(_incident([alert, metric]))

    candidate = _audit_candidate(result, "namespace-metric")
    assert candidate["observable_features"]["anomalous_metric"] is True
    assert candidate["observable_features"]["linked_to_active_alert"] is False
    assert "explicit_active_alert_link" not in candidate["protection_reasons"]


def test_short_metric_name_substring_does_not_create_alert_link() -> None:
    alert = _alert(
        "substring-alert",
        "DatabaseFailure",
        "datacenter",
        severity="warning",
    )
    metric = _metric(
        "substring-metric",
        "a",
        "datacenter",
        [1.0, 1.0],
        baseline=1.0,
    )
    result = _service(
        {"substring-alert": 0.2, "substring-metric": 0.1}, budget=2
    ).infer(_incident([alert, metric]))

    candidate = _audit_candidate(result, "substring-metric")
    assert candidate["observable_features"]["linked_to_active_alert"] is False


def test_alert_schema_field_names_do_not_create_metric_links() -> None:
    alert = _alert(
        "schema-token-alert",
        "DatabaseFailure",
        "datacenter",
        severity="warning",
    )
    metric = _metric(
        "schema-token-metric",
        "unit.type",
        "datacenter",
        [1.0, 1.0],
        baseline=1.0,
    )

    result = _service(
        {"schema-token-alert": 0.1, "schema-token-metric": 1.0}, budget=2
    ).infer(_incident([alert, metric]))

    candidate = _audit_candidate(result, "schema-token-metric")
    assert candidate["observable_features"]["linked_to_active_alert"] is False
    assert candidate["protected"] is False
    assert "explicit_active_alert_link" not in candidate["protection_reasons"]


def test_metric_linkage_does_not_cross_join_two_unrelated_alerts() -> None:
    name_alert = _alert(
        "name-alert",
        "CPUUtilizationHigh",
        "server-1",
        severity="warning",
    )
    entity_alert = _alert(
        "entity-alert",
        "DatabaseFailure",
        "server-2",
        severity="warning",
    )
    metric = _metric(
        "cross-join-metric",
        "cpu.utilization",
        "server-2",
        [10.0, 10.0],
        baseline=10.0,
    )
    result = _service(
        {"name-alert": 0.2, "entity-alert": 0.1, "cross-join-metric": 1.0},
        budget=3,
    ).infer(_incident([name_alert, entity_alert, metric]))

    candidate = _audit_candidate(result, "cross-join-metric")
    assert candidate["observable_features"]["linked_to_active_alert"] is False
    assert candidate["protected"] is False


def test_correlation_link_requires_compatible_entity_and_operational_state() -> None:
    correlated_alert = _alert(
        "correlated-alert",
        "DatabaseFailure",
        "server-1",
        severity="warning",
        correlation="shared-correlation",
    )
    unrelated_alert = _alert(
        "unrelated-entity-alert",
        "QueueFailure",
        "server-2",
        severity="warning",
    )
    metric = _metric(
        "correlated-wrong-entity-metric",
        "cache.hit_ratio",
        "server-2",
        [99.0, 99.0],
        baseline=99.0,
        correlation="shared-correlation",
    )
    result = _service(
        {
            "correlated-alert": 0.2,
            "unrelated-entity-alert": 0.1,
            "correlated-wrong-entity-metric": 1.0,
        },
        budget=3,
    ).infer(_incident([correlated_alert, unrelated_alert, metric]))

    candidate = _audit_candidate(result, "correlated-wrong-entity-metric")
    assert candidate["observable_features"]["linked_to_active_alert"] is False
    assert candidate["alert_linkage_strength"] == "no_match"
    assert "explicit_active_alert_link" not in candidate["protection_reasons"]
    assert candidate["protected"] is False
    assert candidate["observable_features"]["target_ids"] == []
    assert candidate["observable_features"]["target_bearing"] is False


def test_unrelated_protected_facts_are_not_labeled_as_corroboration() -> None:
    scheduler = _alert(
        "unrelated-protected-scheduler",
        "SchedulerAPILatencyHigh",
        "scheduler",
        subsystem="control_plane",
    )
    thermal = _alert(
        "unrelated-protected-thermal",
        "ThermalSensorBiased",
        "rack-r1-row1-03",
        subsystem="thermal",
    )
    service = StateBundleInference(
        _ScoresByIdModel(
            {
                "unrelated-protected-scheduler": 0.5,
                "unrelated-protected-thermal": 0.4,
            }
        ),
        StateBundleInferenceConfig(
            top_m=2,
            anchor_token_budget=10,
            total_token_budget=10,
        ),
        token_estimator=_RowsPlusGroupCount(),
    )

    result = service.infer(_incident([scheduler, thermal]))

    assert _selected_ids(result) == {
        "unrelated-protected-scheduler",
        "unrelated-protected-thermal",
    }
    assert len(result.groups) == 2
    assert all(not group.evidence for group in result.groups)
    assert result.audit_dict()["protected_anchor_deferrals"] == []


def test_unrelated_protected_overflow_cannot_reenter_as_ordinary_support() -> None:
    scheduler = _alert(
        "overflow-protected-scheduler",
        "SchedulerAPILatencyHigh",
        "scheduler",
        subsystem="control_plane",
    )
    thermal = _alert(
        "overflow-protected-thermal",
        "ThermalSensorBiased",
        "rack-r1-row1-03",
        subsystem="thermal",
    )
    service = StateBundleInference(
        _ScoresByIdModel(
            {
                "overflow-protected-scheduler": 0.5,
                "overflow-protected-thermal": 0.4,
            }
        ),
        StateBundleInferenceConfig(
            top_m=2,
            anchor_token_budget=3,
            total_token_budget=3,
            max_anchors=2,
            supports_per_anchor=2,
            minimum_support_score=-100.0,
            minimum_budget_fill_score=-100.0,
        ),
        token_estimator=_RowsPlusGroupCount(),
    )

    result = service.infer(_incident([scheduler, thermal]))

    assert _selected_ids(result) == {"overflow-protected-scheduler"}
    omitted = _audit_candidate(result, "overflow-protected-thermal")
    assert omitted["selected_as"] == "rejected"
    assert omitted["rejection_reason"] == "protected_budget_overflow"
    assert omitted["eviction_reason"] is None
    protected_omissions = result.audit_dict()["protected_omissions"]
    assert len(protected_omissions) == 1
    omission = protected_omissions[0]
    assert omission["observation_id"] == "overflow-protected-thermal"
    assert omission["reason"] == "protected_budget_overflow"
    assert omission["protection_reasons"] == [
        "active_severe_alert",
        "active_target_alert",
    ]
    assert omission["protection_reason"] == "active_severe_alert"
    assert omission["equivalence_group_id"]
    assert omission["selected_representative"] == "overflow-protected-thermal"
    assert omission["rank"] == 2
    assert omission["estimated_token_cost"] == 1
    assert omission["remaining_budget"] == 1
    assert omission["priority"] == {
        "linkage_strength": "no_match",
        "estimated_target_role": "direct_target_candidate",
        "severity": "critical",
        "target_specific": True,
        "recency": QUERY_TIME,
        "diagnostic_score": pytest.approx(2.05),
    }
    assert all(not group.evidence for group in result.groups)
    assert result.audit_dict()["unused_tokens"] == 1
    assert result.audit_dict()["unused_capacity_reason"] == "no_candidate_fits"


def test_protected_repack_can_pivot_on_a_related_selected_support() -> None:
    alert_a = _alert(
        "pivot-alert-a",
        "AlertA",
        "entity-x",
        subsystem="sub-a",
    )
    alert_b = _alert(
        "pivot-alert-b",
        "AlertB",
        "entity-x",
        subsystem="sub-b",
    )
    alert_b["metadata"]["entities"].append(
        {
            "entity_id": "entity-y",
            "role": "affected",
            "confidence": 1.0,
            "provenance": "canonical_fixture",
        }
    )
    alert_c = _alert(
        "pivot-alert-c",
        "AlertC",
        "entity-y",
        subsystem="sub-c",
    )
    service = StateBundleInference(
        _ScoresByIdModel(
            {"pivot-alert-a": 3.0, "pivot-alert-b": 2.0, "pivot-alert-c": 1.0}
        ),
        StateBundleInferenceConfig(
            top_m=3,
            anchor_token_budget=4,
            total_token_budget=4,
        ),
        token_estimator=_RowsPlusGroupCount(),
    )

    result = service.infer(_incident([alert_a, alert_b, alert_c]))

    assert _selected_ids(result) == {
        "pivot-alert-a",
        "pivot-alert-b",
        "pivot-alert-c",
    }
    assert len(result.groups) == 1
    assert result.groups[0].anchor.observation.observation_id == "pivot-alert-b"
    assert {item.observation_id for item in result.groups[0].evidence} == {
        "pivot-alert-a",
        "pivot-alert-c",
    }
    assert result.audit_dict()["protected_omissions"] == []
    assert result.audit_dict()["protected_group_repacks"][-1]["anchor"] == (
        "pivot-alert-b"
    )


def test_coverage_eviction_reconciles_promoted_anchor_diagnostics() -> None:
    static = _config(
        "coverage-static-anchor",
        "simulation.auto_advance",
        False,
        operation="state",
        change_time=0.0,
    )
    support = _log("coverage-log-support", event_type="secondary_event")
    anomaly = _metric(
        "coverage-anomaly",
        "rack.cpu_utilization",
        "rack-r1-row1-01",
        [10.0, 80.0],
        baseline=10.0,
    )
    service = _service(
        {
            "coverage-static-anchor": 20.0,
            "coverage-log-support": 0.5,
            "coverage-anomaly": -10.0,
        },
        budget=2,
        max_anchors=1,
        supports_per_anchor=1,
        minimum_support_score=-100.0,
    )

    result = service.infer(_incident([anomaly, support, static]))

    assert _selected_ids(result) == {"coverage-log-support", "coverage-anomaly"}
    promoted = _audit_candidate(result, "coverage-log-support")
    assert promoted["selected_as"] == "anchor"
    assert promoted["selection_phase"] == "corroboration"
    evicted = _audit_candidate(result, "coverage-static-anchor")
    assert evicted["rejection_reason"] == "no_candidate_fits"
    assert evicted["eviction_reason"] == "evicted_for_coverage:anomalous_metric"


def test_coverage_guard_uses_shared_support_envelope_before_eviction() -> None:
    static = _config(
        "coverage-envelope-static",
        "simulation.auto_advance",
        False,
        operation="state",
        change_time=0.0,
    )
    support = _log("coverage-envelope-log", event_type="secondary_event")
    anomaly = _metric(
        "coverage-envelope-anomaly",
        "rack.cpu_utilization",
        "rack-r1-row1-01",
        [10.0, 80.0],
        baseline=10.0,
    )
    config = StateBundleInferenceConfig(
        top_m=3,
        anchor_token_budget=4,
        total_token_budget=4,
        max_anchors=1,
        supports_per_anchor=1,
        minimum_support_score=-100.0,
        minimum_budget_fill_score=-100.0,
        ann_projection_count=2,
        ann_candidate_multiplier=2,
    )
    service = StateBundleInference(
        _ScoresByIdModel(
            {
                "coverage-envelope-static": 20.0,
                "coverage-envelope-log": 0.5,
                "coverage-envelope-anomaly": -10.0,
            }
        ),
        config,
        token_estimator=_RowsPlusGroupCount(),
    )

    result = service.infer(_incident([anomaly, support, static]))

    assert _selected_ids(result) == {
        "coverage-envelope-static",
        "coverage-envelope-log",
        "coverage-envelope-anomaly",
    }
    assert len(result.groups) == 1
    assert result.used_tokens == result.token_budget == 4
    assert result.audit_dict()["coverage_evictions"] == []
    assert result.audit_dict()["coverage_fallbacks"] == ["anomalous_metric"]


def test_coverage_eviction_skips_sole_prior_category_representative() -> None:
    warning = _alert(
        "coverage-warning",
        "GenericWarning",
        "unknown",
        severity="warning",
    )
    distractor = _log("coverage-distractor", event_type="rare_distractor")
    anomaly = _metric(
        "coverage-second-category",
        "rack.cpu_utilization",
        "rack-r1-row1-01",
        [10.0, 80.0],
        baseline=10.0,
    )
    service = _service(
        {
            "coverage-warning": 0.0,
            "coverage-distractor": 10.0,
            "coverage-second-category": -10.0,
        },
        budget=2,
        max_anchors=2,
        supports_per_anchor=1,
        minimum_support_score=100.0,
    )

    result = service.infer(_incident([warning, distractor, anomaly]))

    assert _selected_ids(result) == {
        "coverage-warning",
        "coverage-second-category",
    }
    assert result.audit_dict()["coverage_evictions"] == [
        {
            "observation_id": "coverage-distractor",
            "reason": "evicted_for_coverage:anomalous_metric",
        }
    ]


def test_unfit_earlier_coverage_class_does_not_block_later_feasible_class() -> None:
    oversized_alert = _alert(
        "coverage-oversized-alert",
        "RackTrafficDrop",
        "rack-r1-row1-01",
    )
    ordinary = _log("coverage-full-budget-log", event_type="rare_distractor")
    anomaly = _metric(
        "coverage-feasible-anomaly",
        "rack.cpu_utilization",
        "rack-r1-row1-02",
        [10.0, 80.0],
        baseline=10.0,
    )
    service = _service(
        {
            "coverage-oversized-alert": -100.0,
            "coverage-full-budget-log": 10.0,
            "coverage-feasible-anomaly": -10.0,
        },
        budget=2,
        costs={
            "coverage-oversized-alert": 3,
            "coverage-full-budget-log": 2,
            "coverage-feasible-anomaly": 1,
        },
        max_anchors=1,
        supports_per_anchor=1,
        minimum_support_score=100.0,
    )

    result = service.infer(_incident([oversized_alert, ordinary, anomaly]))

    assert _selected_ids(result) == {"coverage-feasible-anomaly"}
    assert result.audit_dict()["coverage_fallbacks"] == ["anomalous_metric"]
    assert result.audit_dict()["coverage_evictions"] == [
        {
            "observation_id": "coverage-full-budget-log",
            "reason": "evicted_for_coverage:anomalous_metric",
        }
    ]
    assert result.audit_dict()["protected_omissions"][0]["observation_id"] == (
        "coverage-oversized-alert"
    )


def test_budget_fill_can_readmit_coverage_victim_when_it_still_fits() -> None:
    warning = _alert(
        "readmit-warning",
        "GenericWarning",
        "unknown",
        severity="warning",
    )
    small = _log("readmit-small-log", event_type="small_diagnostic")
    large = _log("readmit-large-log", event_type="large_diagnostic")
    anomaly = _metric(
        "readmit-anomaly",
        "rack.cpu_utilization",
        "rack-r1-row1-01",
        [10.0, 80.0],
        baseline=10.0,
    )
    service = _service(
        {
            "readmit-warning": 0.0,
            "readmit-small-log": 1.0,
            "readmit-large-log": 2.0,
            "readmit-anomaly": -10.0,
        },
        budget=10,
        costs={
            "readmit-warning": 4,
            "readmit-small-log": 1,
            "readmit-large-log": 5,
            "readmit-anomaly": 5,
        },
        max_anchors=2,
        supports_per_anchor=1,
        minimum_support_score=100.0,
        minimum_budget_fill_score=-100.0,
    )

    result = service.infer(_incident([warning, small, large, anomaly]))

    assert _selected_ids(result) == {
        "readmit-warning",
        "readmit-small-log",
        "readmit-anomaly",
    }
    assert result.used_tokens == result.token_budget == 10
    small_audit = _audit_candidate(result, "readmit-small-log")
    assert small_audit["selection_phase"] == "budget_fill"
    assert small_audit["eviction_reason"] is None
    assert result.audit_dict()["coverage_evictions"] == [
        {
            "observation_id": "readmit-large-log",
            "reason": "evicted_for_coverage:anomalous_metric",
        }
    ]


def test_unresolved_slo_and_post_action_metrics_survive_partial_mitigation() -> None:
    pre = [
        _alert(
            "alert-app-pre",
            "ApplicationErrorRateElevated",
            "workload",
            fingerprint="application-errors:default",
            subsystem="application",
        ),
        _alert(
            "alert-sla-pre",
            "sla_violation",
            "datacenter",
            fingerprint="sla:datacenter",
            subsystem="operations",
            details={"health_violation": 1},
        ),
        _metric(
            "metric-error-pre",
            "workload.error_rate",
            "workload",
            [0.0, 35.0],
            baseline=0.0,
            subsystem="application",
            unit="percent",
        ),
        _metric(
            "metric-dropped-pre",
            "workload.dropped_requests",
            "workload",
            [0.0, 700.0],
            baseline=0.0,
            subsystem="application",
            unit="requests/second",
        ),
    ]
    post = [
        _alert(
            "alert-app-post",
            "ApplicationErrorRateElevated",
            "workload",
            fingerprint="application-errors:default",
            subsystem="application",
            start=96.0,
            end=101.0,
        ),
        _alert(
            "alert-sla-post",
            "sla_violation",
            "datacenter",
            fingerprint="sla:datacenter",
            subsystem="operations",
            details={"health_violation": 1},
            start=96.0,
            end=101.0,
        ),
        _metric(
            "metric-error-post",
            "workload.error_rate",
            "workload",
            [35.0, 35.0],
            baseline=0.0,
            start=97.0,
            end=101.0,
            subsystem="application",
            unit="percent",
        ),
        _metric(
            "metric-dropped-post",
            "workload.dropped_requests",
            "workload",
            [700.0, 350.0],
            baseline=0.0,
            start=97.0,
            end=101.0,
            subsystem="application",
            unit="requests/second",
        ),
        _config(
            "config-action-post",
            "controls.action-0001.status",
            "applied",
            operation="set",
            change_time=101.0,
            scope="action-0001",
            subsystem="operations",
        ),
    ]
    scores = {
        **{item["observation_id"]: 0.05 for item in pre},
        **{item["observation_id"]: 0.01 for item in post},
        "config-action-post": 0.99,
    }
    service = _service(scores, budget=8, post_action_retention_cuts=2)
    service.infer(_incident(pre, cut_index=1))

    result = service.infer(
        _incident(post, query_time=101.0, cut_index=2),
        action={
            "action_type": "throttle_workload",
            "parameters": {
                "request_rate_per_second": 1000,
                "tenant_id": "default",
            },
        },
    )

    current_required = {
        "alert-app-post",
        "alert-sla-post",
        "metric-error-post",
        "metric-dropped-post",
    }
    retained_error_id = (
        "retained:pre-action-1:metric:workload.error_rate:workload:metric-error-pre"
    )
    assert current_required | {retained_error_id} <= _selected_ids(result)
    retained_candidate = _audit_candidate(result, retained_error_id)
    assert retained_candidate["protection_reasons"] == ["retained_pre_action_evidence"]
    assert (
        result.audit_dict()["causal_cut_transition"]["unresolved_alerts_retained"]
        is True
    )
    assert (
        result.audit_dict()["causal_cut_transition"]["unresolved_slo_evidence_retained"]
        is True
    )
    for observation_id in ("metric-error-post", "metric-dropped-post"):
        candidate = _audit_candidate(result, observation_id)
        assert candidate["post_action"] is True
        assert "post_action_evaluation" in candidate["protection_reasons"]
    assert (
        result.audit_dict()["causal_cut_transition"]["recorded_actions"][-1][
            "action_type"
        ]
        == "throttle_workload"
    )
    _assert_budget(result)

    service.infer(_incident(post, query_time=102.0, cut_index=3))
    within_window = service.infer(_incident(post, query_time=103.0, cut_index=4))
    assert retained_error_id in _selected_ids(within_window)

    after_window = service.infer(_incident(post, query_time=104.0, cut_index=5))
    assert not any(
        observation_id.startswith("retained:pre-action-1:")
        for observation_id in _selected_ids(after_window)
    )
    assert any(
        item["reason"] == "post_action_effect_evaluated"
        for item in after_window.audit_dict()["causal_cut_transition"][
            "memory_evictions"
        ]
    )
    _assert_budget(after_window)


def test_unrelated_traces_do_not_displace_post_action_evaluation_metrics() -> None:
    alert_app = _alert(
        "trace-pressure-app-alert",
        "ApplicationErrorRateElevated",
        "workload",
        fingerprint="trace-pressure-app",
        subsystem="application",
    )
    alert_sla = _alert(
        "trace-pressure-sla-alert",
        "sla_violation",
        "datacenter",
        fingerprint="trace-pressure-sla",
        details={"health_violation": 1},
    )
    error = _metric(
        "trace-pressure-error",
        "workload.error_rate",
        "workload",
        [35.0, 35.0],
        baseline=0.0,
        subsystem="application",
    )
    dropped = _metric(
        "trace-pressure-dropped",
        "workload.dropped_requests",
        "workload",
        [700.0, 350.0],
        baseline=0.0,
        subsystem="application",
    )
    traces = [_trace(f"trace-pressure-noise-{index}") for index in range(4)]
    for trace in traces:
        trace["payload"].update(
            {
                "operation": "inventory_refresh",
                "source": "inventory",
                "destination": "catalog",
                "status_counts": {"ok": 10.0},
                "retry_count": 0.0,
                "critical_path": False,
            }
        )
    scores = {
        "trace-pressure-app-alert": 0.0,
        "trace-pressure-sla-alert": 0.0,
        "trace-pressure-error": 0.0,
        "trace-pressure-dropped": 0.0,
        **{f"trace-pressure-noise-{index}": 10.0 for index in range(4)},
    }
    service = _service(scores, budget=4)

    result = service.infer(
        _incident([*traces, dropped, error, alert_sla, alert_app]),
        action={
            "action_type": "throttle_workload",
            "parameters": {"request_rate_per_second": 1_000},
        },
    )

    assert _selected_ids(result) == {
        "trace-pressure-app-alert",
        "trace-pressure-sla-alert",
        "trace-pressure-error",
        "trace-pressure-dropped",
    }
    for trace in traces:
        candidate = _audit_candidate(result, trace["observation_id"])
        assert candidate["observable_features"]["action_relevant_trace"] is False
        assert candidate["post_action"] is False
        assert "post_action_evaluation" not in candidate["protection_reasons"]


@pytest.mark.parametrize("healthy_status", ("200", "201", "204", "2xx"))
def test_successful_http_trace_is_not_post_action_protected(
    healthy_status: str,
) -> None:
    trace = _trace(f"healthy-http-{healthy_status}")
    trace["payload"].update(
        {
            "operation": "inventory_refresh",
            "source": "inventory",
            "destination": "catalog",
            "status_counts": {healthy_status: 10.0},
            "retry_count": 0.0,
            "critical_path": False,
        }
    )
    metric = _metric(
        f"healthy-http-evaluation-{healthy_status}",
        "workload.error_rate",
        "workload",
        [20.0, 10.0],
        baseline=0.0,
    )
    result = _service(
        {
            trace["observation_id"]: 10.0,
            metric["observation_id"]: 0.0,
        },
        budget=1,
    ).infer(
        _incident([trace, metric]),
        action={
            "action_type": "throttle_workload",
            "parameters": {"request_rate_per_second": 1_000},
        },
    )

    assert _selected_ids(result) == {metric["observation_id"]}
    trace_audit = _audit_candidate(result, trace["observation_id"])
    assert trace_audit["observable_features"]["action_relevant_trace"] is False
    assert trace_audit["post_action"] is False


def test_resolved_evidence_is_released_after_stable_resolution() -> None:
    fingerprint = "service-capacity:workload"
    firing = _alert(
        "alert-firing",
        "ServiceCapacityDrop",
        "workload",
        fingerprint=fingerprint,
        details={"health_violation": 1},
    )
    resolved_once = _alert(
        "alert-resolved-1",
        "ServiceCapacityDrop",
        "workload",
        fingerprint=fingerprint,
        status="resolved",
        start=96.0,
        end=101.0,
    )
    resolved_twice = _alert(
        "alert-resolved-2",
        "ServiceCapacityDrop",
        "workload",
        fingerprint=fingerprint,
        status="resolved",
        start=96.0,
        end=102.0,
    )
    distractor_1 = _log(
        "log-after-resolution-1", event_type="rare_controller_failure", end=101.0
    )
    distractor_2 = _log(
        "log-after-resolution-2", event_type="rare_controller_failure", end=102.0
    )
    scores = {
        "alert-firing": 0.01,
        "alert-resolved-1": 0.01,
        "alert-resolved-2": 0.01,
        "log-after-resolution-1": 0.99,
        "log-after-resolution-2": 0.99,
    }
    service = _service(
        scores,
        budget=1,
        resolution_stability_cuts=2,
        retention_max_cuts=6,
    )
    first = service.infer(_incident([firing], cut_index=1))
    second = service.infer(
        _incident([resolved_once, distractor_1], query_time=101.0, cut_index=2)
    )
    third = service.infer(
        _incident([resolved_twice, distractor_2], query_time=102.0, cut_index=3)
    )

    assert _selected_ids(first) == {"alert-firing"}
    assert _selected_ids(second) == {"alert-resolved-1"}
    assert _selected_ids(third) == {"log-after-resolution-2"}
    assert any(
        item["reason"] == "condition_resolved_stably"
        for item in third.audit_dict()["causal_cut_transition"]["memory_evictions"]
    )
    _assert_budget(third)


def test_high_scoring_resolved_row_is_not_re_retained_after_release() -> None:
    fingerprint = "released-capacity:workload"
    firing = _alert(
        "released-firing",
        "ServiceCapacityDrop",
        "workload",
        fingerprint=fingerprint,
    )
    resolved_one = _alert(
        "released-resolved-1",
        "ServiceCapacityDrop",
        "workload",
        fingerprint=fingerprint,
        status="resolved",
        end=101.0,
    )
    resolved_two = _alert(
        "released-resolved-2",
        "ServiceCapacityDrop",
        "workload",
        fingerprint=fingerprint,
        status="resolved",
        end=102.0,
    )
    final_log = _log(
        "released-final-log", event_type="rare_controller_failure", end=103.0
    )
    service = _service(
        {
            "released-firing": 0.01,
            "released-resolved-1": 2.0,
            "released-resolved-2": 2.0,
            "released-final-log": 0.01,
        },
        budget=1,
        resolution_stability_cuts=2,
    )

    service.infer(_incident([firing], cut_index=1))
    service.infer(_incident([resolved_one], query_time=101.0, cut_index=2))
    release = service.infer(_incident([resolved_two], query_time=102.0, cut_index=3))
    after = service.infer(_incident([final_log], query_time=103.0, cut_index=4))

    assert _selected_ids(release) == {"released-resolved-2"}
    assert any(
        item["reason"] == "condition_resolved_stably"
        for item in release.audit_dict()["causal_cut_transition"]["memory_evictions"]
    )
    assert _selected_ids(after) == {"released-final-log"}


def test_alternate_request_on_same_snapshot_does_not_advance_retention() -> None:
    """A drill-down view is not a new causal cut."""

    fingerprint = "service-capacity:workload"
    firing = _alert(
        "same-cut-firing",
        "ServiceCapacityDrop",
        "workload",
        fingerprint=fingerprint,
    )
    resolved = _alert(
        "same-cut-resolved",
        "ServiceCapacityDrop",
        "workload",
        fingerprint=fingerprint,
        status="resolved",
        start=96.0,
        end=101.0,
    )
    distractor = _log("same-cut-log", event_type="rare_controller_failure", end=101.0)
    service = _service(
        {
            "same-cut-firing": 0.01,
            "same-cut-resolved": 0.01,
            "same-cut-log": 0.99,
        },
        budget=1,
        resolution_stability_cuts=2,
    )
    service.infer(_incident([firing], cut_index=1))
    second_cut = _incident([resolved, distractor], query_time=101.0, cut_index=2)

    overview = service.infer(second_cut)
    drilldown = service.infer(
        second_cut,
        request=AgentObservationRequest(detail="raw"),
    )

    assert _selected_ids(overview) == {"same-cut-resolved"}
    assert _selected_ids(drilldown) == {"same-cut-resolved"}
    transition = drilldown.audit_dict()["causal_cut_transition"]
    assert transition["cut_index"] == 2
    assert transition["causal_cut_advanced"] is False
    assert not any(
        item["reason"] == "condition_resolved_stably"
        for item in transition["memory_evictions"]
    )


def test_standalone_selector_does_not_retain_across_incident_ids() -> None:
    alert = _alert("episode-a-alert", "RackTrafficDrop", "rack-r1-row1-01")
    log = _log("episode-b-log", event_type="rare_controller_failure")
    service = _service({"episode-a-alert": 0.01, "episode-b-log": 0.99}, budget=1)

    service.infer(_incident([alert], incident_id="episode-a"))
    second = service.infer(_incident([log], incident_id="episode-b"))

    assert _selected_ids(second) == {"episode-b-log"}
    assert second.audit_dict()["causal_cut_transition"]["cut_index"] == 1


def test_budget_fill_skips_oversized_candidate_and_admits_later_fit() -> None:
    anchor = _log("log-anchor", event_type="rare_anchor_event")
    expensive = _config(
        "config-expensive",
        "controls.scheduler.max_retries",
        8,
        previous_value=4,
        operation="update",
        change_time=99.0,
        scope="scheduler",
    )
    cheap = _trace("trace-cheap")
    service = _service(
        {"log-anchor": 1.0, "config-expensive": 0.95, "trace-cheap": 0.80},
        budget=2,
        costs={"log-anchor": 1, "config-expensive": 2, "trace-cheap": 1},
        anchor_token_budget=1,
        max_anchors=1,
        supports_per_anchor=1,
    )

    result = service.infer(_incident([expensive, cheap, anchor]))

    assert _selected_ids(result) == {"log-anchor", "trace-cheap"}
    assert result.used_tokens == result.token_budget == 2
    assert _audit_candidate(result, "config-expensive")["rejection_reason"] == (
        "no_candidate_fits"
    )
    assert result.audit_dict()["unused_capacity_reason"] == "budget_reached"
    _assert_budget(result)


def test_budget_fill_recomputes_and_orders_by_marginal_utility() -> None:
    anchor = _log("marginal-anchor", event_type="rare_anchor_event")
    same_channel = _log("marginal-same-log", event_type="secondary_event")
    cross_domain = _trace(
        "marginal-cross-trace",
        source="tenant:default",
        destination="service:web",
    )
    service = _service(
        {
            "marginal-anchor": 1.0,
            "marginal-same-log": 0.8,
            "marginal-cross-trace": 0.5,
        },
        budget=2,
        anchor_token_budget=1,
        max_anchors=1,
        supports_per_anchor=1,
        minimum_support_score=10.0,
    )

    result = service.infer(_incident([same_channel, cross_domain, anchor]))

    assert _selected_ids(result) == {"marginal-anchor", "marginal-cross-trace"}
    rejected = _audit_candidate(result, "marginal-same-log")
    assert rejected["rejection_reason"] == "no_candidate_fits"
    assert result.used_tokens == result.token_budget


@pytest.mark.parametrize("budget", [1, 2, 3])
def test_selection_never_exceeds_budget(budget: int) -> None:
    observations = [
        _log(f"log-budget-{index}", event_type=f"rare_event_{index}")
        for index in range(3)
    ]
    scores = {
        item["observation_id"]: 1.0 - index * 0.1
        for index, item in enumerate(observations)
    }
    service = _service(scores, budget=budget, max_anchors=3)

    result = service.infer(_incident(observations))

    assert result.used_tokens == budget
    _assert_budget(result)


def test_default_selector_records_exact_serialized_budget_utilization() -> None:
    alert = _alert(
        "exact-accounting-alert",
        "SchedulerAPILatencyHigh",
        "scheduler",
    )
    service = StateBundleInference(
        _ScoresByIdModel({"exact-accounting-alert": 0.1}),
        StateBundleInferenceConfig(top_m=1),
    )

    result = service.infer(_incident([alert]))
    audit = result.audit_dict()

    assert audit["token_accounting_exact"] is True
    assert audit["token_budget_semantics"] == "serialized_tokens"
    assert audit["budget_accounted_telemetry_tokens"] == result.used_tokens
    assert audit["actual_serialized_telemetry_tokens"] == result.used_tokens
    assert audit["actual_serialized_within_budget"] is True
    assert audit["unused_tokens"] == result.token_budget - result.used_tokens
    assert audit["utilization_ratio"] == pytest.approx(
        result.used_tokens / result.token_budget
    )


def test_alternate_compact_tokenizer_is_not_labeled_as_benchmark_exact() -> None:
    changed = _config(
        "alternate-tokenizer-config",
        "controls.scheduler.retry_limit",
        4,
        previous_value=2,
        operation="update",
        change_time=99.0,
        scope="scheduler",
    )
    service = StateBundleInference(
        _ScoresByIdModel({"alternate-tokenizer-config": 0.5}),
        StateBundleInferenceConfig(top_m=1),
        token_estimator=CompactAgentTokenEstimator("cl100k_base"),
    )

    result = service.infer(_incident([changed]))
    public = result.to_dict()
    observations = [
        row
        for group in public["evidence_groups"]
        for row in (
            group["anchor"],
            *group["corroborating_observations"],
        )
    ]
    references = [
        {
            "group": index,
            "anchor": group["anchor"]["observation_id"],
            "supports": [
                row["observation_id"] for row in group["corroborating_observations"]
            ],
        }
        for index, group in enumerate(public["evidence_groups"])
    ]
    benchmark_exact = compact_statebundle_bundle_token_cost(
        observations,
        references,
        query_time_seconds=QUERY_TIME,
        target_scope_ambiguity=public["target_scope_ambiguity"],
        target_scope_candidates=public["target_scope_candidates"],
    )
    audit = result.audit_dict()

    assert audit["token_accounting_exact"] is False
    assert audit["token_accounting_encoding"] == "cl100k_base"
    assert audit["actual_serialization_encoding"] == "o200k_base"
    assert audit["actual_serialized_telemetry_tokens"] == benchmark_exact


@pytest.mark.parametrize(
    ("cost", "expected_reason"),
    [(1, "all_candidates_exhausted"), (4, "no_candidate_fits")],
)
def test_unused_budget_has_explicit_reason(cost: int, expected_reason: str) -> None:
    observation = _log("log-underfill", event_type="rare_underfill_event")
    service = _service(
        {"log-underfill": 1.0},
        budget=3,
        costs={"log-underfill": cost},
        anchor_token_budget=3,
    )

    result = service.infer(_incident([observation]))

    assert result.audit_dict()["unused_capacity_reason"] == expected_reason
    assert result.audit_dict()["unused_tokens"] == 3 - result.used_tokens
    assert result.audit_dict()["unused_tokens"] > 0
    _assert_budget(result)


def test_unused_budget_records_when_no_candidate_has_positive_utility() -> None:
    observation = _log("log-below-utility", event_type="routine_notice")
    service = _service(
        {"log-below-utility": 0.0},
        budget=3,
        minimum_anchor_gain=10.0,
        minimum_budget_fill_score=10.0,
    )

    result = service.infer(_incident([observation]))

    assert result.used_tokens == 0
    assert result.audit_dict()["unused_tokens"] == 3
    assert result.audit_dict()["unused_capacity_reason"] == "no_useful_candidate"
    assert (
        _audit_candidate(result, "log-below-utility")["rejection_reason"]
        == "below_budget_fill_utility"
    )


def test_resolved_alerts_do_not_displace_post_action_recovery_metrics() -> None:
    resolved_alerts = [
        _alert(
            f"resolved-post-action-{index}",
            "OldCriticalCondition",
            f"rack-r1-row1-{index + 1:02d}",
            status="resolved",
            severity="critical",
        )
        for index in range(2)
    ]
    error_rate = _metric(
        "post-action-error-rate",
        "workload.error_rate",
        "workload",
        [35.0, 20.0],
        baseline=0.0,
        subsystem="application",
    )
    dropped = _metric(
        "post-action-dropped",
        "workload.dropped_requests",
        "workload",
        [700.0, 300.0],
        baseline=0.0,
        subsystem="application",
    )
    scores = {
        **{item["observation_id"]: 10.0 for item in resolved_alerts},
        "post-action-error-rate": 0.0,
        "post-action-dropped": 0.0,
    }

    result = _service(scores, budget=2).infer(
        _incident([*resolved_alerts, error_rate, dropped]),
        action={
            "action_type": "throttle_workload",
            "parameters": {"request_rate_per_second": 1_000},
        },
    )

    assert _selected_ids(result) == {
        "post-action-error-rate",
        "post-action-dropped",
    }
    for alert in resolved_alerts:
        candidate = _audit_candidate(result, alert["observation_id"])
        assert candidate["post_action"] is False
        assert "post_action_evaluation" not in candidate["protection_reasons"]


def test_zero_normal_comparisons_are_kept_only_as_bounded_representatives() -> None:
    target_alert = _alert(
        "packet-loss-alert",
        "PacketLossHigh",
        "rack-r1-row1-01",
        severity="warning",
        subsystem="network",
    )
    target_metric = _metric(
        "packet-loss-target",
        "rack.packet_loss",
        "rack-r1-row1-01",
        [0.0, 20.0],
        baseline=0.0,
        subsystem="network",
    )
    peers = [
        _metric(
            f"packet-loss-normal-{index}",
            "rack.packet_loss",
            f"rack-r1-row1-{index + 2:02d}",
            [0.0, 0.0],
            baseline=0.0,
            subsystem="network",
        )
        for index in range(6)
    ]
    scores = {
        "packet-loss-alert": 0.0,
        "packet-loss-target": 0.0,
        **{item["observation_id"]: 10.0 - index for index, item in enumerate(peers)},
    }
    result = _service(
        scores,
        budget=5,
        zero_series_representatives=2,
        semantic_group_representatives=2,
    ).infer(_incident([*peers, target_metric, target_alert]))

    selected_peers = _selected_ids(result) & {item["observation_id"] for item in peers}
    assert len(selected_peers) == 2
    assert {
        _audit_candidate(result, item["observation_id"])["rejection_reason"]
        for item in peers
        if item["observation_id"] not in selected_peers
    } == {"equivalence_group_cap"}


def test_log_retention_identity_includes_visible_entity_scope() -> None:
    server_a = _log(
        "entity-log-a",
        event_type="request_error",
        entity_id="server-a",
    )
    server_b = _log(
        "entity-log-b",
        event_type="request_error",
        entity_id="server-b",
        end=101.0,
    )
    for log in (server_a, server_b):
        log["payload"]["template_id"] = "T123"
        log["payload"]["template"] = "request_error on {entity}"
    service = _service({"entity-log-a": 1.0, "entity-log-b": 1.0}, budget=2)
    service.infer(_incident([server_a], cut_index=1))

    result = service.infer(
        _incident([server_b], query_time=101.0, cut_index=2),
    )

    assert _selected_ids(result) == {"entity-log-a", "entity-log-b"}
    current = _audit_candidate(result, "entity-log-b")
    assert current["sticky"] is False
    assert current["logical_key"].endswith(":server-b")


def test_action_relevant_log_support_is_retained_for_pre_action_comparison() -> None:
    changed = _config(
        "action-log-config",
        "controls.scheduler.retry_limit",
        4,
        previous_value=2,
        operation="update",
        change_time=99.0,
        scope="scheduler",
    )
    log = _log(
        "action-log-support",
        event_type="request_error",
        subsystem="application",
        entity_id="workload",
    )
    post = _metric(
        "action-log-post-metric",
        "workload.error_rate",
        "workload",
        [20.0, 10.0],
        baseline=0.0,
    )
    service = _service(
        {
            "action-log-config": 10.0,
            "action-log-support": 1.0,
            "action-log-post-metric": 0.1,
        },
        budget=4,
        max_anchors=1,
        supports_per_anchor=1,
        minimum_support_score=-100.0,
    )
    first = service.infer(_incident([log, changed], cut_index=1))
    assert _audit_candidate(first, "action-log-support")["selected_as"] == "support"

    second = service.infer(
        _incident([post], query_time=101.0, cut_index=2),
        action={
            "action_type": "throttle_workload",
            "parameters": {"request_rate_per_second": 1_000},
        },
    )

    retained_pre_action = {
        item
        for item in _selected_ids(second)
        if item.startswith("retained:pre-action-1:")
    }
    assert len(retained_pre_action) == 1
    assert next(iter(retained_pre_action)).endswith(":action-log-support")


def test_observation_order_change_does_not_create_a_new_causal_cut() -> None:
    first_log = _log("stable-cut-a", event_type="request_error", entity_id="server-a")
    second_log = _log("stable-cut-b", event_type="queue_warning", entity_id="server-b")
    service = _service({"stable-cut-a": 1.0, "stable-cut-b": 0.9}, budget=2)
    service.infer(_incident([first_log, second_log], cut_index=1))

    result = service.infer(
        _incident([second_log, first_log], cut_index=1),
        request=AgentObservationRequest(channels=("log",), log_limit=2),
    )

    transition = result.audit_dict()["causal_cut_transition"]
    assert transition["causal_cut_advanced"] is False
    assert transition["cut_index"] == 1


def test_integer_causal_watermark_cannot_move_backwards() -> None:
    observation = _log("integer-watermark-log", event_type="request_error")

    def incident(marker: int) -> CanonicalIncident:
        copied = deepcopy(observation)
        copied["metadata"]["available_at_sequence"] = marker
        return CanonicalIncident.from_dict(
            {
                "schema_version": CANONICAL_SCHEMA_VERSION,
                "episode_id": "integer-watermark-episode",
                "snapshot_id": f"integer-watermark-{marker}",
                "query_time_seconds": QUERY_TIME,
                "query_watermark_sequence": marker,
                "observations": [copied],
            }
        )

    service = _service({"integer-watermark-log": 1.0}, budget=1)
    service.infer(incident(2))
    with pytest.raises(ValueError, match="watermark must not move backwards"):
        service.infer(incident(1))


def test_targeted_alert_subsystem_and_lookback_constraints_are_enforced() -> None:
    requested_alert = _alert(
        "requested-scheduler-alert",
        "SchedulerAPILatencyHigh",
        "scheduler",
        subsystem="control_plane",
    )
    unrelated = _config(
        "unrelated-high-score-config",
        "rack.health.enabled",
        True,
        operation="state",
    )
    alert_result = _service(
        {
            "requested-scheduler-alert": 0.0,
            "unrelated-high-score-config": 10.0,
        },
        budget=1,
    ).infer(
        _incident([unrelated, requested_alert]),
        request=AgentObservationRequest(
            channels=("alert",),
            subsystem_ids=("control_plane",),
            alert_names=("SchedulerAPILatencyHigh",),
        ),
    )
    assert _selected_ids(alert_result) == {"requested-scheduler-alert"}
    assert alert_result.to_dict()["request_status"]["status"] == "satisfied"

    old_metric = _metric(
        "outside-lookback",
        "scheduler.pending_operations",
        "scheduler",
        [100.0, 100.0],
        baseline=0.0,
        start=45.0,
        end=50.0,
        subsystem="control_plane",
    )
    recent_metric = _metric(
        "inside-lookback",
        "scheduler.pending_operations",
        "scheduler",
        [10.0, 20.0],
        baseline=0.0,
        start=90.0,
        end=100.0,
        subsystem="control_plane",
    )
    metric_result = _service(
        {"outside-lookback": 10.0, "inside-lookback": 0.0}, budget=1
    ).infer(
        _incident([old_metric, recent_metric]),
        request=AgentObservationRequest(
            channels=("metric",),
            metric_names=("scheduler.pending_operations",),
            entity_ids=("scheduler",),
            lookback_seconds=20.0,
        ),
    )
    assert _selected_ids(metric_result) == {"inside-lookback"}
    assert (
        _audit_candidate(metric_result, "outside-lookback")["rejection_reason"]
        == "outside_request_scope"
    )


def test_direct_action_context_accepts_non_oracle_operational_split() -> None:
    service = _service({"action-split-log": 1.0}, budget=1)
    result = service.infer(
        _incident([_log("action-split-log")]),
        action={
            "action_type": "rebalance_traffic",
            "parameters": {"split": 0.5, "relevance_score": 0.8},
        },
    )

    action = result.audit_dict()["causal_cut_transition"]["recorded_actions"][-1]
    assert action["parameters"] == {"split": 0.5, "relevance_score": 0.8}


def test_inference_is_repeat_deterministic_and_rejects_hidden_context() -> None:
    observations = [
        _alert("alert-safe", "RackTrafficDrop", "rack-r1-row1-01", severity="warning"),
        _metric(
            "metric-safe",
            "rack.cpu_utilization",
            "rack-r1-row1-01",
            [8.0, 0.0],
            baseline=8.0,
        ),
        _config(
            "config-safe",
            "controls.scheduler.max_retries",
            4,
            previous_value=2,
            operation="update",
            change_time=99.0,
        ),
        _log("log-safe"),
        _trace("trace-safe"),
    ]
    scores = {item["observation_id"]: 0.5 for item in observations}
    incident = _incident(observations)
    service = _service(scores, budget=5)

    first = service.infer(incident)
    repeated = service.infer(incident)
    independent = _service(scores, budget=5).infer(incident)

    assert first.to_json() == repeated.to_json() == independent.to_json()
    assert first.audit_dict() == repeated.audit_dict() == independent.audit_dict()
    assert "selection_audit" not in first.to_dict()
    assert not (FORBIDDEN_VISIBLE_KEYS & _all_keys(first.to_dict()))
    assert service.model.seen_batch_types == [IncidentBatch]
    _assert_budget(first)

    with pytest.raises(ValueError, match="non-inference action field"):
        _service(scores, budget=5).infer(
            incident,
            action={
                "action_type": "throttle_workload",
                "parameters": {"success_criteria": {"error_rate_max": 0.0}},
            },
        )
    with pytest.raises(ValueError, match="non-inference action field"):
        _service(scores, budget=5).infer(
            incident,
            action={
                "action_type": "throttle_workload",
                "parameters": {"ground-truth": "oracle-secret"},
            },
        )
    with pytest.raises(ValueError, match="non-inference action field"):
        _service(scores, budget=5).infer(
            incident,
            action={
                "action_type": "throttle_workload",
                "parameters": {"root/cause": "oracle-secret"},
            },
        )

    poisoned = _log("log-poisoned")
    poisoned["payload"]["root_cause"] = "oracle-secret"
    with pytest.raises(ValueError, match="training-only field"):
        _incident([poisoned])


@pytest.mark.parametrize(
    "oracle_key",
    (
        "expected_diagnosis",
        "expected.mitigation",
        "evaluator-state",
        "oracle",
        "solution",
        "action_result",
        "active_faults_after",
    ),
)
def test_direct_inference_rejects_nested_oracle_action_aliases(
    oracle_key: str,
) -> None:
    with pytest.raises(ValueError, match="non-inference action field"):
        _service({"oracle-action-log": 1.0}, budget=1).infer(
            _incident([_log("oracle-action-log")]),
            action={
                "action_type": "throttle_workload",
                "parameters": {"nested": {oracle_key: "oracle-secret"}},
            },
        )


def test_legitimate_observable_phase_and_background_payloads_remain_valid() -> None:
    log = _log("log-valid-phase", event_type="deployment_notice")
    log["payload"]["phase"] = "startup"
    log["payload"]["background"] = "controller-maintenance"

    result = _service({"log-valid-phase": 0.5}, budget=1).infer(_incident([log]))
    payload = result.to_dict()["evidence_groups"][0]["anchor"]["payload"]

    assert payload["phase"] == "startup"
    assert payload["background"] == "controller-maintenance"


def test_action_sequence_remains_monotonic_when_audit_history_is_bounded() -> None:
    observations = [
        _metric(
            f"sequence-metric-{index}",
            "workload.error_rate",
            "workload",
            [float(index), float(index + 1)],
            baseline=0.0,
            start=96.0,
            end=100.0 + index,
        )
        for index in range(5)
    ]
    scores = {item["observation_id"]: 0.5 for item in observations}
    service = _service(scores, budget=8, retention_max_cuts=2)
    service.infer(_incident([observations[0]], cut_index=1))
    result = None
    for index in range(1, 5):
        result = service.infer(
            _incident(
                [observations[index]],
                query_time=100.0 + index,
                cut_index=index + 1,
            ),
            action={
                "action_type": "throttle_workload",
                "parameters": {"request_rate_per_second": 1000 - index},
            },
        )

    assert result is not None
    sequences = [
        item["sequence"]
        for item in result.audit_dict()["causal_cut_transition"]["recorded_actions"]
    ]
    assert sequences == [3, 4]


def test_control_plane_evidence_is_not_displaced_by_zero_rack_metrics() -> None:
    alert = _alert(
        "control-alert",
        "SchedulerAPILatencyHigh",
        "scheduler",
        subsystem="control_plane",
        threshold={"latency_ms": 20.0},
        details={"latency_ms": 76.0},
    )
    scheduler_latency = _metric(
        "control-scheduler-latency",
        "control_plane.scheduler_api_latency",
        "scheduler",
        [7.65, 7.65, 7.65, 76.0],
        baseline=7.65,
        subsystem="control_plane",
        unit="milliseconds",
    )
    pending = _metric(
        "control-pending",
        "workload.queue_length",
        "workload",
        [0.0, 0.0, 0.0, 22.0],
        baseline=0.0,
        subsystem="workload",
        unit="requests",
    )
    capacity = _metric(
        "control-capacity",
        "workload.service_capacity",
        "workload",
        [24000.0, 24000.0, 24000.0, 5640.0],
        baseline=24000.0,
        subsystem="workload",
        unit="requests/second",
    )
    zeros = [
        _metric(
            f"control-zero-{index}",
            "rack.failed_server_count",
            f"rack-r1-row{1 + index // 3}-{1 + index % 3:02d}",
            [0.0, 0.0, 0.0, 0.0],
            baseline=0.0,
            subsystem="compute",
        )
        for index in range(6)
    ]
    static = _config(
        "control-static",
        "simulation.auto_advance",
        False,
        operation="set",
        change_time=0.0,
    )
    scores = {
        "control-alert": 0.01,
        "control-scheduler-latency": 0.02,
        "control-pending": 0.01,
        "control-capacity": 0.01,
        "control-static": 0.95,
        **{f"control-zero-{index}": 0.99 for index in range(6)},
    }
    service = _service(scores, budget=4)

    result = service.infer(
        _incident([static, *zeros, capacity, pending, scheduler_latency, alert])
    )
    selected = _selected_ids(result)

    assert {"control-alert", "control-scheduler-latency"} <= selected
    assert len(selected & {f"control-zero-{index}" for index in range(6)}) <= (
        service.config.zero_series_representatives
    )
    assert (
        _audit_candidate(result, "control-scheduler-latency")["observable_features"][
            "anomalous_metric"
        ]
        is True
    )
    _assert_budget(result)


@pytest.mark.parametrize("fault", ["network", "thermal"])
def test_target_bearing_network_or_thermal_evidence_survives(fault: str) -> None:
    if fault == "network":
        target_id = "rack-r1-row1-01"
        root_alert = _alert(
            "network-root-alert",
            "RackTrafficDrop",
            target_id,
            severity="warning",
            subsystem="network",
        )
        root_metric = _metric(
            "network-root-metric",
            "rack.cpu_utilization",
            target_id,
            [8.33, 8.33, 8.33, 0.0],
            baseline=8.33,
            subsystem="network",
            unit="percent",
        )
        comparison = _metric(
            "network-comparison",
            "rack.cpu_utilization",
            "rack-r1-row1-02",
            [10.0, 10.0, 10.0, 10.0],
            baseline=10.0,
            subsystem="network",
            unit="percent",
        )
        global_alerts: list[dict[str, Any]] = []
    else:
        target_id = "rack-r1-row1-03"
        root_alert = _alert(
            "thermal-root-alert",
            "thermal_sensor_health",
            target_id,
            subsystem="thermal",
            details={
                "inlet_temperature_c": 21.53,
                "reported_inlet_temperature_c": 34.13,
                "temperature_sensor_disagreement_c": 12.6,
                "temperature_sensor_status": "biased",
            },
        )
        root_metric = _metric(
            "thermal-root-metric",
            "rack.temperature_sensor_disagreement",
            target_id,
            [0.0, 0.0, 0.0, 12.6],
            baseline=0.0,
            subsystem="thermal",
            unit="degC",
        )
        comparison = _metric(
            "thermal-comparison",
            "rack.temperature_sensor_disagreement",
            "rack-r1-row1-02",
            [0.0, 0.0, 0.0, 0.0],
            baseline=0.0,
            subsystem="thermal",
            unit="degC",
        )
        global_alerts = [
            _alert(
                "thermal-global-sla",
                "sla_violation",
                "datacenter",
                subsystem="operations",
                details={"thermal_critical": 1},
            )
        ]
    zeros = [
        _metric(
            f"{fault}-zero-{index}",
            "rack.health_flapping_count",
            f"rack-r1-row2-{index + 1:02d}",
            [0.0, 0.0, 0.0, 0.0],
            baseline=0.0,
            subsystem="compute",
        )
        for index in range(3)
    ]
    observations = [*zeros, comparison, root_metric, *global_alerts, root_alert]
    scores = {
        item["observation_id"]: (
            0.01
            if item["observation_id"]
            in {root_alert["observation_id"], root_metric["observation_id"]}
            else 0.99
        )
        for item in observations
    }
    service = _service(scores, budget=4)

    result = service.infer(_incident(observations))

    assert {
        root_alert["observation_id"],
        root_metric["observation_id"],
    } <= _selected_ids(result)
    assert (
        _audit_candidate(result, root_alert["observation_id"])["observable_features"][
            "target_bearing"
        ]
        is True
    )
    assert (
        _audit_candidate(result, root_metric["observation_id"])["observable_features"][
            "anomalous_metric"
        ]
        is True
    )
    _assert_budget(result)


def test_insufficient_application_mitigation_does_not_replace_evidence_with_zero_health() -> (
    None
):
    pre_alert = _alert(
        "mitigation-app-pre",
        "ApplicationErrorRateElevated",
        "workload",
        fingerprint="application-errors:default",
        subsystem="application",
    )
    post_alert = _alert(
        "mitigation-app-post",
        "ApplicationErrorRateElevated",
        "workload",
        fingerprint="application-errors:default",
        subsystem="application",
        start=96.0,
        end=101.0,
    )
    pre_sla = _alert(
        "mitigation-sla-pre",
        "sla_violation",
        "datacenter",
        fingerprint="sla:datacenter",
        details={"health_violation": 1},
    )
    post_sla = _alert(
        "mitigation-sla-post",
        "sla_violation",
        "datacenter",
        fingerprint="sla:datacenter",
        details={"health_violation": 1},
        start=96.0,
        end=101.0,
    )
    pre_error = _metric(
        "mitigation-error-pre",
        "workload.error_rate",
        "workload",
        [0.0, 35.0],
        baseline=0.0,
        subsystem="application",
        unit="percent",
    )
    post_error = _metric(
        "mitigation-error-post",
        "workload.error_rate",
        "workload",
        [35.0, 35.0],
        baseline=0.0,
        start=97.0,
        end=101.0,
        subsystem="application",
        unit="percent",
    )
    post_dropped = _metric(
        "mitigation-dropped-post",
        "workload.dropped_requests",
        "workload",
        [700.0, 350.0],
        baseline=0.0,
        start=97.0,
        end=101.0,
        subsystem="application",
        unit="requests/second",
    )
    post_zeros = [
        _metric(
            f"mitigation-zero-{index}",
            "rack.health_flapping_count",
            f"rack-r1-row2-{index + 1:02d}",
            [0.0, 0.0, 0.0, 0.0],
            baseline=0.0,
            start=97.0,
            end=101.0,
            subsystem="compute",
        )
        for index in range(3)
    ]
    scores = {
        "mitigation-app-pre": 0.05,
        "mitigation-sla-pre": 0.05,
        "mitigation-error-pre": 0.05,
        "mitigation-app-post": 0.01,
        "mitigation-sla-post": 0.01,
        "mitigation-error-post": 0.01,
        "mitigation-dropped-post": 0.01,
        **{f"mitigation-zero-{index}": 0.99 for index in range(3)},
    }
    service = _service(scores, budget=7)
    service.infer(_incident([pre_alert, pre_sla, pre_error], cut_index=1))

    result = service.infer(
        _incident(
            [*post_zeros, post_dropped, post_error, post_sla, post_alert],
            query_time=101.0,
            cut_index=2,
        ),
        action={
            "action_type": "throttle_workload",
            "parameters": {
                "request_rate_per_second": 1000,
                "tenant_id": "default",
            },
        },
    )
    selected = _selected_ids(result)

    assert {
        "mitigation-app-post",
        "mitigation-sla-post",
        "mitigation-error-post",
        "mitigation-dropped-post",
    } <= selected
    assert len(selected & {f"mitigation-zero-{index}" for index in range(3)}) <= 1
    assert (
        result.audit_dict()["causal_cut_transition"]["unresolved_alerts_retained"]
        is True
    )
    _assert_budget(result)


def test_stable_slo_release_cannot_be_undone_by_same_cut_targeted_view() -> None:
    active = _metric(
        "slo-active",
        "rack.failed_server_count",
        "rack-r1-row1-01",
        [0.0, 1.0],
        baseline=0.0,
    )
    resolved_once = _metric(
        "slo-resolved-1",
        "rack.failed_server_count",
        "rack-r1-row1-01",
        [0.0, 0.0],
        baseline=0.0,
        end=101.0,
    )
    resolved_twice = _metric(
        "slo-resolved-2",
        "rack.failed_server_count",
        "rack-r1-row1-01",
        [0.0, 0.0],
        baseline=0.0,
        end=102.0,
    )
    benign = _log("after-stable-release", event_type="routine_notice", end=103.0)
    service = _service(
        {
            "slo-active": 1.0,
            "slo-resolved-1": 1.0,
            "slo-resolved-2": 1.0,
            "after-stable-release": 10.0,
        },
        budget=1,
        resolution_stability_cuts=2,
    )
    service.infer(_incident([active], cut_index=1))
    service.infer(_incident([resolved_once], query_time=101.0, cut_index=2))
    release_cut = _incident([resolved_twice], query_time=102.0, cut_index=3)
    released = service.infer(release_cut)
    logical_key = "metric:rack.failed_server_count:rack-r1-row1-01"
    assert any(
        item["logical_key"] == logical_key
        and item["reason"] == "condition_resolved_stably"
        for item in released.audit_dict()["causal_cut_transition"]["memory_evictions"]
    )

    targeted_same_cut = service.infer(
        release_cut,
        request=AgentObservationRequest(
            channels=("metric",),
            detail="raw",
            metric_names=("rack.failed_server_count",),
            entity_ids=("rack-r1-row1-01",),
        ),
    )
    assert _selected_ids(targeted_same_cut) == {"slo-resolved-2"}
    assert logical_key not in service._memory.retained
    assert targeted_same_cut.audit_dict()["causal_cut_transition"][
        "stable_resolution_tombstones"
    ] == [logical_key]

    next_cut = service.infer(_incident([benign], query_time=103.0, cut_index=4))
    assert _selected_ids(next_cut) == {"after-stable-release"}
    assert not any(item["retained"] for item in next_cut.audit_dict()["candidates"])


def test_action_scope_prefers_direct_cooling_effects_over_unrelated_health_names() -> (
    None
):
    supply = _metric(
        "cooling-supply-post",
        "cooling.supply_air_temperature",
        "cooling-unit-1",
        [18.0, 16.0],
        baseline=18.0,
        subsystem="cooling",
    )
    fan = _metric(
        "cooling-fan-post",
        "cooling.fan_speed",
        "cooling-unit-1",
        [50.0, 90.0],
        baseline=50.0,
        subsystem="cooling",
    )
    storage_error = _metric(
        "unrelated-storage-error",
        "storage.error_rate",
        "storage",
        [1.0, 1.0],
        baseline=1.0,
        subsystem="storage",
    )
    scheduler_latency = _metric(
        "unrelated-scheduler-latency",
        "control_plane.scheduler_api_latency",
        "scheduler",
        [10.0, 10.0],
        baseline=10.0,
        subsystem="control_plane",
    )
    cooling_bookkeeping = _log(
        "unrelated-cooling-bookkeeping",
        event_type="routine_notice",
        subsystem="cooling",
        entity_id="cooling-control-plane",
    )
    result = _service(
        {
            "cooling-supply-post": 0.0,
            "cooling-fan-post": 0.0,
            "unrelated-storage-error": 10.0,
            "unrelated-scheduler-latency": 9.0,
            "unrelated-cooling-bookkeeping": 11.0,
        },
        budget=2,
    ).infer(
        _incident([storage_error, scheduler_latency, cooling_bookkeeping, supply, fan]),
        action={
            "action_type": "set_cooling",
            "parameters": {
                "target": "cooling-unit-1",
                "fan_speed_percent": 90,
                "supply_air_temperature_c": 16,
            },
        },
    )

    assert _selected_ids(result) == {"cooling-supply-post", "cooling-fan-post"}
    for observation_id in ("cooling-supply-post", "cooling-fan-post"):
        assert _audit_candidate(result, observation_id)["post_action"] is True
    for observation_id in (
        "unrelated-storage-error",
        "unrelated-scheduler-latency",
        "unrelated-cooling-bookkeeping",
    ):
        candidate = _audit_candidate(result, observation_id)
        assert candidate["post_action"] is False
        assert "post_action_evaluation" not in candidate["protection_reasons"]


def test_cooling_action_preserves_direct_before_and_after_metrics() -> None:
    pre_supply = _metric(
        "cooling-supply-pre",
        "cooling.supply_air_temperature",
        "cooling-unit-1",
        [18.0, 18.0],
        baseline=18.0,
        subsystem="cooling",
    )
    pre_fan = _metric(
        "cooling-fan-pre",
        "cooling.fan_speed",
        "cooling-unit-1",
        [50.0, 50.0],
        baseline=50.0,
        subsystem="cooling",
    )
    post_supply = _metric(
        "cooling-supply-after",
        "cooling.supply_air_temperature",
        "cooling-unit-1",
        [18.0, 16.0],
        baseline=18.0,
        end=101.0,
        subsystem="cooling",
    )
    post_fan = _metric(
        "cooling-fan-after",
        "cooling.fan_speed",
        "cooling-unit-1",
        [50.0, 90.0],
        baseline=50.0,
        end=101.0,
        subsystem="cooling",
    )
    service = _service(
        {
            "cooling-supply-pre": 0.5,
            "cooling-fan-pre": 0.5,
            "cooling-supply-after": 0.1,
            "cooling-fan-after": 0.1,
        },
        budget=4,
    )
    service.infer(_incident([pre_supply, pre_fan], cut_index=1))

    result = service.infer(
        _incident([post_supply, post_fan], query_time=101.0, cut_index=2),
        action={
            "action_type": "set_cooling",
            "parameters": {
                "target": "cooling-unit-1",
                "fan_speed_percent": 90,
                "supply_air_temperature_c": 16,
            },
        },
    )

    assert {"cooling-supply-after", "cooling-fan-after"} <= _selected_ids(result)
    retained_pre = {
        observation_id
        for observation_id in _selected_ids(result)
        if observation_id.startswith("retained:pre-action-1:")
    }
    assert len(retained_pre) == 2
    assert any(item.endswith(":cooling-supply-pre") for item in retained_pre)
    assert any(item.endswith(":cooling-fan-pre") for item in retained_pre)


def test_known_action_effect_support_is_available_for_later_pre_action_snapshot() -> (
    None
):
    anchor_log = _log("cooling-support-anchor", event_type="routine_notice")
    before = _metric(
        "cooling-support-before",
        "cooling.supply_air_temperature",
        "cooling-unit-1",
        [18.0, 18.0],
        baseline=18.0,
        subsystem="cooling",
    )
    after = _metric(
        "cooling-support-after",
        "cooling.supply_air_temperature",
        "cooling-unit-1",
        [18.0, 16.0],
        baseline=18.0,
        end=101.0,
        subsystem="cooling",
    )
    service = _service(
        {
            "cooling-support-anchor": 10.0,
            "cooling-support-before": 0.1,
            "cooling-support-after": 0.1,
        },
        budget=3,
        max_anchors=1,
        supports_per_anchor=1,
        minimum_support_score=-100.0,
    )
    first = service.infer(_incident([before, anchor_log], cut_index=1))
    assert _audit_candidate(first, "cooling-support-before")["selected_as"] == (
        "support"
    )

    result = service.infer(
        _incident([after], query_time=101.0, cut_index=2),
        action={
            "action_type": "set_cooling",
            "parameters": {
                "target": "cooling-unit-1",
                "supply_air_temperature_c": 16,
            },
        },
    )
    assert "cooling-support-after" in _selected_ids(result)
    assert any(
        observation_id.startswith("retained:pre-action-1:")
        and observation_id.endswith(":cooling-support-before")
        for observation_id in _selected_ids(result)
    )


def test_overlapping_actions_keep_each_unexpired_evaluation_scope() -> None:
    cooling = _metric(
        "overlap-cooling-temperature",
        "rack.inlet_temperature",
        "rack-07",
        [31.0, 25.0],
        baseline=22.0,
        subsystem="thermal",
    )
    errors = _metric(
        "overlap-workload-errors",
        "workload.error_rate",
        "tenant-blue",
        [20.0, 10.0],
        baseline=0.0,
        subsystem="application",
    )
    dropped = _metric(
        "overlap-workload-dropped",
        "workload.dropped_requests",
        "tenant-blue",
        [500.0, 200.0],
        baseline=0.0,
        subsystem="application",
    )
    distractor = _config(
        "overlap-static-config",
        "storage.replication_factor",
        3,
        operation="snapshot",
        change_time=1.0,
        scope="storage",
        subsystem="storage",
    )
    service = _service(
        {
            "overlap-cooling-temperature": -5.0,
            "overlap-workload-errors": -4.0,
            "overlap-workload-dropped": -3.0,
            "overlap-static-config": 20.0,
        },
        budget=3,
        retention_max_cuts=1,
        post_action_retention_cuts=3,
    )

    service.infer(
        _incident([cooling], cut_index=1),
        action={
            "action_type": "set_cooling",
            "parameters": {"target": "cooling-unit-1", "fan_speed_percent": 90},
        },
    )
    result = service.infer(
        _incident(
            [cooling, errors, dropped, distractor],
            query_time=101.0,
            cut_index=2,
        ),
        action={
            "action_type": "throttle_workload",
            "parameters": {"tenant_id": "tenant-blue", "request_rate": 1000},
        },
    )

    assert _selected_ids(result) == {
        "overlap-cooling-temperature",
        "overlap-workload-errors",
        "overlap-workload-dropped",
    }
    for observation_id in (
        "overlap-cooling-temperature",
        "overlap-workload-errors",
        "overlap-workload-dropped",
    ):
        candidate = _audit_candidate(result, observation_id)
        assert candidate["post_action"] is True
        assert "post_action_evaluation" in candidate["protection_reasons"]
    transition = result.audit_dict()["causal_cut_transition"]
    assert transition["active_action_sequences"] == [1, 2]
    assert [item["sequence"] for item in transition["recorded_actions"]] == [2]
    assert [
        item["expires_cut"] for item in transition["active_evaluation_actions"]
    ] == [4, 5]


def test_application_sla_linkage_prefers_direct_counters_over_zero_rack_fanout() -> (
    None
):
    """Reproduce the application-error protected-fanout policy artifact.

    A generic SLA alert may describe a workload counter without sharing the
    workload row's entity ID.  Matching the alert field *and its value/state*
    identifies the direct counterpart; lexical overlap alone must not promote
    every healthy per-rack series to protected evidence.
    """

    alert = _alert(
        "application-error-alert",
        "ApplicationErrorRateElevated",
        "default",
        subsystem="application",
        details={
            "error_rate_percent": 35.0,
            "dropped_requests_per_second": 700.0,
            "latency_ms": 22.0,
        },
    )
    generic_sla = _alert(
        "generic-sla-alert",
        "sla_violation",
        "datacenter",
        subsystem="operations",
        details={
            "network_error_rate": 0.0,
            "power_budget_violating_racks": 0,
            "workload_error_rate_percent": 35.0,
        },
    )
    error_rate = _metric(
        "workload-error-rate-direct",
        "workload.error_rate",
        "workload",
        [0.0, 0.0, 0.0, 35.0],
        baseline=0.0,
        subsystem="application",
        unit="percent",
    )
    dropped = _metric(
        "workload-dropped-direct",
        "workload.dropped_requests",
        "workload",
        [0.0, 0.0, 0.0, 700.0],
        baseline=0.0,
        subsystem="application",
        unit="requests/second",
    )
    rack_zeros = [
        _metric(
            f"rack-network-zero-{index}",
            "rack.network_error_rate",
            f"rack-r1-row{1 + index // 3}-{1 + index % 3:02d}",
            [0.0, 0.0, 0.0, 0.0],
            baseline=0.0,
            subsystem="compute",
            unit="percent",
        )
        for index in range(6)
    ]
    scores = {
        "application-error-alert": -0.8,
        "generic-sla-alert": -0.6,
        "workload-error-rate-direct": -0.7,
        "workload-dropped-direct": -0.9,
        **{f"rack-network-zero-{index}": 1.0 - index / 100.0 for index in range(6)},
    }
    incident = _incident([*rack_zeros, dropped, error_rate, generic_sla, alert])

    first = _service(
        scores,
        budget=5,
        top_m=2,
        zero_series_representatives=1,
    ).infer(incident)
    independent = _service(
        scores,
        budget=5,
        top_m=2,
        zero_series_representatives=1,
    ).infer(incident)

    selected = _selected_ids(first)
    assert {
        "application-error-alert",
        "generic-sla-alert",
        "workload-error-rate-direct",
        "workload-dropped-direct",
    } <= selected
    assert len(selected & {f"rack-network-zero-{index}" for index in range(6)}) <= 1

    for observation_id, matched_field in (
        ("workload-error-rate-direct", "error_rate_percent"),
        ("workload-dropped-direct", "dropped_requests_per_second"),
    ):
        candidate = _audit_candidate(first, observation_id)
        assert candidate["alert_linkage_strength"] == "direct_counterpart"
        assert candidate["matched_alert_field"] == matched_field
        assert candidate["matched_alert_entity"] == "default"
        assert "explicit_active_alert_link" in candidate["protection_reasons"]
        assert candidate["estimated_target_role"] == "direct_target_candidate"

    zero_audits = [
        _audit_candidate(first, f"rack-network-zero-{index}") for index in range(6)
    ]
    assert all(
        item["alert_linkage_strength"] == "weak_context_match" for item in zero_audits
    )
    assert all(
        item["matched_alert_field"] == "network_error_rate" for item in zero_audits
    )
    assert all(item["matched_alert_entity"] == "datacenter" for item in zero_audits)
    assert all(
        "explicit_active_alert_link" not in item["protection_reasons"]
        for item in zero_audits
    )
    assert len({item["equivalence_group_id"] for item in zero_audits}) == 1
    selected_zero_ids = {
        item["observation_id"]
        for item in zero_audits
        if item["selected_as"] != "rejected"
    }
    assert len(selected_zero_ids) <= 1
    assert all(
        item["equivalence_representative_id"] in selected_zero_ids
        for item in zero_audits
        if selected_zero_ids
    )

    # New policy metadata must be deterministic and remain within the same
    # budget even when safeguards override the adversarial learned top-M.
    assert first.to_json() == independent.to_json()
    assert first.audit_dict() == independent.audit_dict()
    assert first.used_tokens <= first.token_budget == 5
    assert first.audit_dict()["learned_top_m_coverage_misses"]
    assert {
        "workload-error-rate-direct",
        "workload-dropped-direct",
    } & {
        item["observation_id"]
        for item in first.audit_dict()["selected_outside_learned_top_m"]
    }


def test_protected_equivalence_arbitration_is_a_hard_no_reentry_decision() -> None:
    """A protected alias omitted by group arbitration cannot re-enter later."""

    aliases = [
        _alert(
            f"power-alert-alias-{index}",
            "RackPowerBudgetHigh",
            "rack-r1-row1-01",
            start=96.0 + index / 10.0,
            subsystem="power",
            details={"power_budget_utilization": 1.20},
        )
        for index in range(4)
    ]
    scores = {
        "power-alert-alias-0": 0.70,
        "power-alert-alias-1": 0.80,
        "power-alert-alias-2": 0.90,
        "power-alert-alias-3": 0.60,
    }
    result = _service(
        scores,
        budget=8,
        semantic_group_representatives=1,
        minimum_support_score=-100.0,
        minimum_budget_fill_score=-100.0,
    ).infer(_incident(aliases))

    # The ample budget makes this a representative-cap decision, not token
    # overflow.  The highest-priority alias is the one deterministic member.
    assert _selected_ids(result) == {"power-alert-alias-2"}
    audits = {
        item["observation_id"]: item for item in result.audit_dict()["candidates"]
    }
    representatives = {
        item["equivalence_representative_id"] for item in audits.values()
    }
    assert representatives == {"power-alert-alias-2"}
    assert len({item["equivalence_group_id"] for item in audits.values()}) == 1

    omitted_ids = set(audits) - {"power-alert-alias-2"}
    for observation_id in omitted_ids:
        candidate = audits[observation_id]
        assert candidate["protected"] is True
        assert candidate["selected_as"] == "rejected"
        assert candidate["selection_phase"] is None
        assert "equivalence" in candidate["rejection_reason"]

    omission_rows = {
        item["observation_id"]: item
        for item in result.audit_dict()["protected_omissions"]
    }
    assert set(omission_rows) == omitted_ids
    for row in omission_rows.values():
        assert row["protection_reasons"]
        assert row["equivalence_group_id"]
        assert "equivalence" in row["reason"]
        assert row["selected_representative"] == "power-alert-alias-2"
        assert isinstance(row["rank"], int) and row["rank"] >= 1
        assert row["estimated_token_cost"] == 1


def test_control_plane_scope_roles_distinguish_direct_from_downstream_effects() -> None:
    """Reproduce the control-plane/workload target-scope ambiguity."""

    scheduler_alert = _alert(
        "scheduler-latency-alert",
        "SchedulerAPILatencyHigh",
        "scheduler",
        severity="warning",
        subsystem="control_plane",
        details={"scheduler_api_latency_ms": 76.0},
        threshold={"scheduler_api_latency_ms": 20.0},
    )
    scheduler_latency = _metric(
        "scheduler-api-latency-direct",
        "control_plane.scheduler_api_latency",
        "scheduler",
        [4.0, 4.0, 4.0, 76.0],
        baseline=4.0,
        subsystem="control_plane",
        unit="milliseconds",
    )
    pending = _metric(
        "scheduler-pending-direct",
        "control_plane.scheduler_pending_operations",
        "scheduler",
        [0.0, 0.0, 0.0, 22.0],
        baseline=0.0,
        subsystem="control_plane",
        unit="operations",
    )
    workload_latency = _metric(
        "workload-latency-downstream",
        "workload.latency",
        "workload",
        [7.65, 7.65, 7.65, 22.21],
        baseline=7.65,
        subsystem="application",
        unit="milliseconds",
    )
    capacity = _metric(
        "workload-capacity-downstream",
        "workload.service_capacity",
        "workload",
        [24000.0, 24000.0, 24000.0, 5640.0],
        baseline=24000.0,
        subsystem="application",
        unit="requests/second",
    )
    scores = {
        "scheduler-latency-alert": 0.05,
        "scheduler-api-latency-direct": 0.04,
        "scheduler-pending-direct": 0.03,
        # Mirror the policy-v2 artifact: downstream symptoms have higher
        # learned scores than the component-local control-plane evidence.
        "workload-latency-downstream": 0.95,
        "workload-capacity-downstream": 0.90,
    }
    incident = _incident(
        [
            capacity,
            workload_latency,
            pending,
            scheduler_latency,
            scheduler_alert,
        ]
    )
    result = StateBundleInference(
        _ScoresByIdModel(scores),
        StateBundleInferenceConfig(
            top_m=3,
            anchor_token_budget=4_096,
            total_token_budget=4_096,
            max_anchors=5,
            supports_per_anchor=4,
            diversity_weight=0.0,
            minimum_anchor_gain=0.0,
            minimum_support_score=-10.0,
            minimum_budget_fill_score=-10.0,
        ),
    ).infer(incident)

    selected = _selected_ids(result)
    assert {
        "scheduler-latency-alert",
        "scheduler-api-latency-direct",
        "scheduler-pending-direct",
        "workload-latency-downstream",
        "workload-capacity-downstream",
    } <= selected

    for observation_id in (
        "scheduler-latency-alert",
        "scheduler-api-latency-direct",
        "scheduler-pending-direct",
    ):
        assert (
            _audit_candidate(result, observation_id)["estimated_target_role"]
            == "direct_target_candidate"
        )
    for observation_id in (
        "workload-latency-downstream",
        "workload-capacity-downstream",
    ):
        assert (
            _audit_candidate(result, observation_id)["estimated_target_role"]
            == "downstream_affected_scope"
        )

    audit = result.audit_dict()
    assert audit["target_scope_ambiguity"] is True
    scope_candidates = {
        item["scope"]: item for item in audit["target_scope_candidates"]
    }
    assert scope_candidates["control_plane"]["estimated_role"] == (
        "direct_target_candidate"
    )
    assert scope_candidates["workload"]["estimated_role"] == (
        "downstream_affected_scope"
    )
    assert {
        "scheduler-api-latency-direct",
        "scheduler-pending-direct",
    } <= set(scope_candidates["control_plane"]["supporting_observation_ids"])

    # The safe, non-oracle scope summary must reach the controlled agent; an
    # audit-only role label would not address the reported submission error.
    public = result.to_dict()
    assert public["target_scope_ambiguity"] is True
    assert public["target_scope_candidates"] == audit["target_scope_candidates"]
    rendered = render_snapshot(public, condition="statebundle")
    assert rendered["target_scope_ambiguity"] is True
    assert rendered["target_scope_candidates"] == public["target_scope_candidates"]
    assert result.used_tokens <= result.token_budget == 4_096
    assert result.audit_dict()["token_accounting_exact"] is True
    assert rendered["budget"]["within_budget"] is True
    assert rendered["budget"]["actual_serialized_tokens"] == result.used_tokens
    assert not (FORBIDDEN_VISIBLE_KEYS & _all_keys(public))


def test_policy_v2_candidate_audit_separates_learned_rank_from_policy_disposition() -> (
    None
):
    """Candidate diagnostics explain model ranking versus policy displacement."""

    alert = _alert(
        "audit-alert",
        "SLAErrorRateHigh",
        "default",
        subsystem="application",
        details={"error_rate": 20.0},
    )
    direct = _metric(
        "audit-direct",
        "workload.error_rate",
        "workload",
        [0.0, 20.0],
        baseline=0.0,
        subsystem="application",
        unit="percent",
    )
    static = _config(
        "audit-static",
        "simulation.auto_advance",
        False,
        operation="state",
        change_time=0.0,
    )
    result = _service(
        {"audit-alert": 0.1, "audit-direct": 0.0, "audit-static": 1.0},
        budget=2,
        top_m=1,
    ).infer(_incident([static, direct, alert]))

    required = {
        "learned_relevance_score",
        "learned_rank",
        "final_ranking_score",
        "final_rank",
        "learned_top_m",
        "protection_reasons",
        "alert_linkage_strength",
        "alert_linkage_reason",
        "matched_alert_field",
        "matched_alert_entity",
        "equivalence_group_id",
        "equivalence_representative_id",
        "estimated_target_role",
        "selected_as",
        "selection_phase",
        "rejection_reason",
        "estimated_token_cost",
        "remaining_budget_at_decision",
    }
    candidates = result.audit_dict()["candidates"]
    assert candidates
    assert all(required <= set(candidate) for candidate in candidates)
    learned_ranks = sorted(item["learned_rank"] for item in candidates)
    final_ranks = sorted(item["final_rank"] for item in candidates)
    assert learned_ranks == final_ranks == [1, 2, 3]
    assert _audit_candidate(result, "audit-static")["learned_top_m"] is True
    assert _audit_candidate(result, "audit-direct")["learned_top_m"] is False
    assert _audit_candidate(result, "audit-direct")["selected_as"] != "rejected"
    assert result.audit_dict()["learned_top_m_coverage_misses"]
    assert result.audit_dict()["selected_outside_learned_top_m"]
    _assert_budget(result)
