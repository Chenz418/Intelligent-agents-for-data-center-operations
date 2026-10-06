r"""Budgeted StateBundle inference from Appendix D.

The implementation performs observable-only encoding/pooling, top-M relevance
filtering, deterministic greedy quality-diversity anchor selection, projected
approximate-nearest-neighbor support expansion, duplicate suppression, token
accounting, and deterministic agent-safe serialization.

The exact form of the corroboration score :math:`\kappa`, tokenizer, greedy
tie-breaking, and ANN library are not fixed by the paper.  Each is explicit and
replaceable here; the defaults are deterministic and dependency-free beyond
PyTorch.
"""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields as dataclass_fields, replace
from typing import Any, Protocol, cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from aiopslab.agent_telemetry import (
    AgentObservationRequest,
    compact_statebundle_bundle_token_cost,
    compact_statebundle_observation_token_cost,
)

from aiopslab.statebundle.config import StateBundleInferenceConfig
from aiopslab.statebundle.policy import (
    ObservableSignals,
    analyze_observations,
    request_is_targeted,
    reranking_components,
)
from aiopslab.statebundle.types import (
    CanonicalIncident,
    CanonicalObservation,
    CorroboratingEvidenceGroup,
    IncidentBatch,
    RedactionPolicy,
    SelectedAnchor,
    StateBundleOutput,
    assert_observable_only,
)


_FORBIDDEN_ACTION_CONTEXT_KEYS = frozenset(
    {
        "accepted",
        "action_result",
        "action_schema_ref",
        "action_type",
        "active_fault",
        "active_faults",
        "active_faults_after",
        "active_faults_before",
        "associated_fault_effect_ids",
        "available_actions",
        "benchmark_action_coverage",
        "causal_role",
        "channels",
        "detail",
        "effect_id",
        "entity_ids",
        "episode_id",
        "error",
        "evaluator",
        "evaluator_state",
        "expected",
        "expected_diagnosis",
        "expected_mitigation",
        "fault",
        "fault_id",
        "fault_label_id",
        "fault_mechanism",
        "fault_target",
        "fault_type",
        "ground_truth",
        "host_visibility",
        "http_status",
        "include_action_schema",
        "include_config",
        "inference_visible_target",
        "injected_faults",
        "log_limit",
        "lookback_seconds",
        "local_effect_or_anomaly_id",
        "metric_names",
        "observation",
        "oracle",
        "pair_label",
        "prototype_id",
        "propagation_depth",
        "propagation_membership",
        "propagation_path",
        "remaining_duration_seconds",
        "root_cause",
        "scenario",
        "score_hint",
        "score_hints",
        "sim_time_seconds",
        "sim_time_seconds_after",
        "sim_time_seconds_before",
        "solution",
        "step_summary",
        "subsystem_ids",
        "subsystem_supervision",
        "success_criteria",
        "symptom_family",
        "training_labels",
    }
)


class TokenEstimator(Protocol):
    def estimate(
        self, observation: CanonicalObservation, policy: RedactionPolicy
    ) -> int: ...


class BundleTokenEstimator(Protocol):
    def estimate_bundle(
        self,
        groups: Sequence[tuple[CanonicalObservation, Sequence[CanonicalObservation]]],
        policy: RedactionPolicy,
        *,
        query_time_seconds: float,
        request: AgentObservationRequest | None,
        target_scope_ambiguity: bool | None = None,
        target_scope_candidates: Sequence[Mapping[str, Any]] | None = None,
    ) -> int: ...


class _InferenceModelOutput(Protocol):
    aspect_embeddings: Tensor
    relevance_scores: Tensor


@dataclass(frozen=True, slots=True)
class CharacterTokenEstimator:
    """Approximate deterministic fallback for paper token cost ell_i.

    The public/default and production path uses CompactAgentTokenEstimator.
    This estimator remains available for experiments and test injection; its
    accounting units are explicitly marked non-exact in selection diagnostics.
    """

    characters_per_token: float = 4.0

    def __post_init__(self) -> None:
        if isinstance(self.characters_per_token, bool) or not isinstance(
            self.characters_per_token, (int, float)
        ):
            raise TypeError("characters_per_token must be numeric")
        if (
            not math.isfinite(float(self.characters_per_token))
            or self.characters_per_token <= 0
        ):
            raise ValueError("characters_per_token must be finite and positive")

    def estimate(
        self, observation: CanonicalObservation, policy: RedactionPolicy
    ) -> int:
        encoded = json.dumps(
            observation.to_agent_dict(policy),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        return max(1, math.ceil(len(encoded) / self.characters_per_token))

    def estimate_bundle(
        self,
        groups: Sequence[tuple[CanonicalObservation, Sequence[CanonicalObservation]]],
        policy: RedactionPolicy,
        *,
        query_time_seconds: float,
        request: AgentObservationRequest | None,
        target_scope_ambiguity: bool | None = None,
        target_scope_candidates: Sequence[Mapping[str, Any]] | None = None,
    ) -> int:
        del (
            query_time_seconds,
            request,
            target_scope_ambiguity,
            target_scope_candidates,
        )
        seen: set[str] = set()
        total = 0
        for anchor, supports in groups:
            for observation in (anchor, *supports):
                if observation.observation_id in seen:
                    continue
                seen.add(observation.observation_id)
                total += self.estimate(observation, policy)
        return total


@dataclass(frozen=True, slots=True)
class CompactAgentTokenEstimator:
    """Cost the deterministic compact table fragment sent to agents.

    Selection still receives the complete :class:`CanonicalObservation`,
    including source/correlation/quality metadata used by encoders and
    duplicate suppression.  Only the cost projection uses the agent-facing
    serialization. A singleton fragment includes the shared-column header and
    evidence-group association, so summed costs are conservative when several
    rows later share table/group envelopes.
    """

    encoding_name: str = "o200k_base"

    def __post_init__(self) -> None:
        if not isinstance(self.encoding_name, str) or not self.encoding_name.strip():
            raise ValueError("encoding_name must be non-empty")

    def estimate(
        self, observation: CanonicalObservation, policy: RedactionPolicy
    ) -> int:
        return compact_statebundle_observation_token_cost(
            observation.to_agent_dict(policy),
            query_time_seconds=observation.metadata.event_end_time_seconds,
            encoding_name=self.encoding_name,
        )

    def estimate_bundle(
        self,
        groups: Sequence[tuple[CanonicalObservation, Sequence[CanonicalObservation]]],
        policy: RedactionPolicy,
        *,
        query_time_seconds: float,
        request: AgentObservationRequest | None,
        target_scope_ambiguity: bool | None = None,
        target_scope_candidates: Sequence[Mapping[str, Any]] | None = None,
    ) -> int:
        observations: list[Mapping[str, Any]] = []
        references: list[dict[str, Any]] = []
        seen: set[str] = set()
        for group_index, (anchor, supports) in enumerate(groups):
            ordered = (anchor, *supports)
            for observation in ordered:
                if observation.observation_id in seen:
                    continue
                seen.add(observation.observation_id)
                observations.append(observation.to_agent_dict(policy))
            references.append(
                {
                    "group": group_index,
                    "anchor": anchor.observation_id,
                    "supports": [item.observation_id for item in supports],
                }
            )
        return compact_statebundle_bundle_token_cost(
            observations,
            references,
            query_time_seconds=query_time_seconds,
            encoding_name=self.encoding_name,
            request=request,
            target_scope_ambiguity=target_scope_ambiguity,
            target_scope_candidates=target_scope_candidates,
        )


class NeighborIndex(Protocol):
    """Replaceable approximate-nearest-neighbor aspect index."""

    def query(
        self, query_index: int, count: int, *, exclude: frozenset[int] = frozenset()
    ) -> tuple[int, ...]: ...


class ProjectedANNIndex:
    """Random-projection sorted-window ANN with deterministic exact reranking.

    Building P sorted one-dimensional projections costs O(P N log N).  A query
    retrieves small windows from those projections and computes cosine scores
    only for their union, avoiding a dense N-by-N matrix.
    """

    def __init__(
        self,
        vectors: Tensor,
        stable_ids: Sequence[str],
        *,
        projection_count: int,
        candidate_multiplier: int,
        seed: int,
    ):
        if vectors.ndim != 2:
            raise ValueError("ANN vectors must be a [count, dimension] tensor")
        if vectors.shape[0] != len(stable_ids):
            raise ValueError("ANN vector and identifier counts must match")
        if projection_count < 1 or candidate_multiplier < 1:
            raise ValueError("ANN projection and candidate counts must be positive")
        self.vectors = F.normalize(vectors.detach(), dim=-1)
        self.stable_ids = tuple(stable_ids)
        self.projection_count = projection_count
        self.candidate_multiplier = candidate_multiplier
        if not len(stable_ids):
            self.projections = torch.empty(
                (vectors.shape[1], projection_count), device=vectors.device
            )
            self.sorted_values: tuple[Tensor, ...] = ()
            self.sorted_indices: tuple[Tensor, ...] = ()
            return
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        projections = torch.randn(
            vectors.shape[1], projection_count, generator=generator
        ).to(device=vectors.device, dtype=vectors.dtype)
        self.projections = F.normalize(projections, dim=0)
        projected = self.vectors @ self.projections
        values: list[Tensor] = []
        indices: list[Tensor] = []
        for dimension in range(projection_count):
            order = torch.argsort(projected[:, dimension], stable=True)
            indices.append(order)
            values.append(projected[order, dimension])
        self.sorted_values = tuple(values)
        self.sorted_indices = tuple(indices)

    def query(
        self, query_index: int, count: int, *, exclude: frozenset[int] = frozenset()
    ) -> tuple[int, ...]:
        if count <= 0 or not self.stable_ids:
            return ()
        if not 0 <= query_index < len(self.stable_ids):
            raise IndexError(query_index)
        target_candidates = min(
            len(self.stable_ids), max(count, count * self.candidate_multiplier)
        )
        projected_query = self.vectors[query_index] @ self.projections
        per_projection = max(
            2, math.ceil(target_candidates / max(1, self.projection_count))
        )
        candidates: set[int] = set()
        for dimension, (values, indices) in enumerate(
            zip(self.sorted_values, self.sorted_indices)
        ):
            position = int(
                torch.searchsorted(values, projected_query[dimension]).item()
            )
            half = per_projection // 2
            start = max(0, position - half)
            end = min(len(self.stable_ids), start + per_projection)
            start = max(0, end - per_projection)
            candidates.update(int(value) for value in indices[start:end].tolist())

        candidates.difference_update(exclude)
        candidates.discard(query_index)
        # Sparse projection windows can be too small for tiny datasets.  This
        # stable fallback fills only the requested candidate pool, not N^2.
        if len(candidates) < min(target_candidates, len(self.stable_ids) - 1):
            for index in sorted(
                range(len(self.stable_ids)), key=self.stable_ids.__getitem__
            ):
                if index != query_index and index not in exclude:
                    candidates.add(index)
                    if len(candidates) >= target_candidates:
                        break
        similarities = {
            index: float(
                torch.dot(self.vectors[query_index], self.vectors[index]).item()
            )
            for index in candidates
        }
        ordered = sorted(
            candidates,
            key=lambda index: (-similarities[index], self.stable_ids[index]),
        )
        return tuple(ordered[:count])


def _duplicate_signature(
    observation: CanonicalObservation,
    policy: RedactionPolicy,
) -> tuple[Any, ...]:
    """Identify exact duplicates in the configured agent-visible projection."""

    projected = observation.to_agent_dict(policy)
    # IDs are identity, not content. Including observation_id here would make
    # the content fallback incapable of recognizing any duplicate rows with
    # independently assigned canonical IDs.
    projected.pop("observation_id", None)
    payload = json.dumps(
        projected,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return ("agent_visible_content", payload)


def _deduplicate_indices(
    observations: Sequence[CanonicalObservation],
    relevance_scores: Tensor,
    policy: RedactionPolicy | None = None,
) -> tuple[int, ...]:
    active_policy = policy or RedactionPolicy()
    best: dict[tuple[Any, ...], int] = {}
    for index, observation in enumerate(observations):
        signature = _duplicate_signature(observation, active_policy)
        incumbent = best.get(signature)
        if incumbent is None:
            best[signature] = index
            continue
        current_key = (
            float(relevance_scores[index].item()),
            -len(observation.observation_id),
            observation.observation_id,
        )
        incumbent_observation = observations[incumbent]
        incumbent_key = (
            float(relevance_scores[incumbent].item()),
            -len(incumbent_observation.observation_id),
            incumbent_observation.observation_id,
        )
        if current_key > incumbent_key:
            best[signature] = index
    return tuple(
        sorted(best.values(), key=lambda index: observations[index].observation_id)
    )


def _temporal_compatibility(
    left: CanonicalObservation,
    right: CanonicalObservation,
    scale_seconds: float,
) -> float:
    left_start = left.metadata.event_start_time_seconds
    left_end = left.metadata.event_end_time_seconds
    right_start = right.metadata.event_start_time_seconds
    right_end = right.metadata.event_end_time_seconds
    if max(left_start, right_start) <= min(left_end, right_end):
        return 1.0
    gap = max(left_start, right_start) - min(left_end, right_end)
    return math.exp(-gap / scale_seconds)


def _support_score(
    anchor: CanonicalObservation,
    candidate: CanonicalObservation,
    anchor_aspect: Tensor,
    candidate_aspect: Tensor,
    candidate_relevance: float,
    config: StateBundleInferenceConfig,
) -> float:
    aspect = float(
        F.cosine_similarity(
            anchor_aspect.unsqueeze(0), candidate_aspect.unsqueeze(0)
        ).item()
    )
    temporal = _temporal_compatibility(anchor, candidate, config.temporal_scale_seconds)
    subsystem = float(
        anchor.metadata.primary_subsystem == candidate.metadata.primary_subsystem
    )
    cross_modal = float(anchor.channel is not candidate.channel)
    return (
        config.support_aspect_weight * aspect
        + config.support_relevance_weight * candidate_relevance
        + config.support_temporal_weight * temporal
        + config.support_subsystem_weight * subsystem
        + config.cross_modal_bonus * cross_modal
    )


def _assert_agent_safe(value: Any, path: str = "statebundle") -> None:
    try:
        assert_observable_only(value, path)
    except ValueError as error:
        raise RuntimeError(str(error)) from error


@dataclass(slots=True)
class _Candidate:
    observation: CanonicalObservation
    observation_index: int
    aspect: Tensor
    learned_score: float
    signals: ObservableSignals
    estimated_cost: int
    sticky: bool = False
    retained: bool = False
    post_action: bool = False
    redundant: bool = False
    protected_reasons: tuple[str, ...] = ()
    components: tuple[tuple[str, float], ...] = ()
    final_score: float = 0.0
    selected_as: str | None = None
    selection_phase: str | None = None
    rejection_reason: str | None = None
    eviction_reason: str | None = None
    learned_rank: int = 0
    final_rank: int = 0
    learned_top_m: bool = False
    equivalence_representative_id: str | None = None
    equivalence_rank: int | None = None
    hard_excluded: bool = False
    remaining_budget_at_decision: int | None = None

    @property
    def identifier(self) -> str:
        return self.observation.observation_id

    @property
    def protected(self) -> bool:
        return bool(self.protected_reasons)


@dataclass(slots=True)
class _WorkingGroup:
    anchor: _Candidate
    supports: list[_Candidate] = field(default_factory=list)


@dataclass(slots=True)
class _RetainedEvidence:
    memory_key: str
    observation: CanonicalObservation
    aspect: Tensor
    learned_score: float
    signals: ObservableSignals
    estimated_cost: int
    components: tuple[tuple[str, float], ...]
    final_score: float
    admitted_cut: int
    expires_cut: int
    unresolved_kind: str | None = None
    pre_action: bool = False


@dataclass(slots=True)
class _EpisodeMemory:
    active: bool = True
    cut_index: int = 0
    retained: dict[str, _RetainedEvidence] = field(default_factory=dict)
    resolution_streaks: dict[str, int] = field(default_factory=dict)
    stable_resolution_tombstones: dict[str, int] = field(default_factory=dict)
    last_selected_logical_keys: set[str] = field(default_factory=set)
    actions: list[dict[str, Any]] = field(default_factory=list)
    evaluation_actions: list[dict[str, Any]] = field(default_factory=list)
    next_action_sequence: int = 1
    causal_cut_key: tuple[Any, ...] | None = None
    processed_action_fingerprints: set[str] = field(default_factory=set)
    cache_key: tuple[Any, ...] | None = None
    cached_output: StateBundleOutput | None = None


_ACTION_EVIDENCE_TERMS = (
    "alert",
    "capacity",
    "congestion",
    "demand",
    "drop",
    "error",
    "health",
    "latency",
    "queue",
    "request",
    "sla",
    "slo",
    "throughput",
    "traffic",
    "violation",
)
_ACTION_DOMAIN_TERMS: dict[str, frozenset[str]] = {
    "calibrate_sensor": frozenset({"sensor", "thermal", "temperature"}),
    "set_cooling": frozenset({"cooling", "thermal", "rack"}),
    "migrate_workload": frozenset(
        {"workload", "placement", "rack", "application", "network"}
    ),
    "throttle_workload": frozenset(
        {"workload", "application", "network", "load", "balancer"}
    ),
    "update_autoscaler_policy": frozenset(
        {"autoscaler", "capacity", "workload", "application"}
    ),
    "repair_monitoring_pipeline": frozenset(
        {"monitoring", "telemetry", "observability", "sensor"}
    ),
    "update_placement_policy": frozenset({"placement", "workload", "rack", "server"}),
    "update_load_balancer_config": frozenset(
        {"load", "balancer", "application", "backend", "routing"}
    ),
    "set_server_maintenance": frozenset({"maintenance", "server", "rack", "capacity"}),
    "clear_server_maintenance": frozenset(
        {"maintenance", "server", "rack", "capacity"}
    ),
}
_ACTION_EFFECT_TERMS: dict[str, frozenset[str]] = {
    "calibrate_sensor": frozenset(
        {
            "sensor",
            "calibration",
            "offset",
            "temperature",
            "reported",
            "actual",
            "health",
            "trusted",
        }
    ),
    "set_cooling": frozenset(
        {"cooling", "fan", "speed", "supply", "air", "temperature", "thermal", "health"}
    ),
    "migrate_workload": frozenset(
        {
            "workload",
            "allocation",
            "placement",
            "demand",
            "utilization",
            "traffic",
            "latency",
            "congestion",
        }
    ),
    "throttle_workload": frozenset(
        {
            "workload",
            "demand",
            "request",
            "rate",
            "throughput",
            "congestion",
            "latency",
            "error",
            "drop",
            "dropped",
            "queue",
        }
    ),
    "update_autoscaler_policy": frozenset(
        {
            "autoscaler",
            "capacity",
            "replica",
            "utilization",
            "pending",
            "cooldown",
            "workload",
        }
    ),
    "repair_monitoring_pipeline": frozenset(
        {
            "monitoring",
            "telemetry",
            "ingest",
            "availability",
            "sensor",
            "health",
            "delay",
            "missing",
        }
    ),
    "update_placement_policy": frozenset(
        {
            "placement",
            "strategy",
            "rack",
            "server",
            "allocation",
            "imbalance",
            "violation",
            "capacity",
        }
    ),
    "update_load_balancer_config": frozenset(
        {
            "load",
            "balancer",
            "backend",
            "routing",
            "weight",
            "request",
            "latency",
            "error",
            "drop",
        }
    ),
    "set_server_maintenance": frozenset(
        {"maintenance", "server", "rack", "available", "capacity", "failed", "health"}
    ),
    "clear_server_maintenance": frozenset(
        {"maintenance", "server", "rack", "available", "capacity", "failed", "health"}
    ),
}
_ACTION_VERB_TERMS = frozenset(
    {"set", "update", "repair", "calibrate", "migrate", "throttle", "clear"}
)
_KNOWN_ACTION_EVIDENCE_TERMS = frozenset(_ACTION_EVIDENCE_TERMS).union(
    *(terms for terms in _ACTION_DOMAIN_TERMS.values()),
    *(terms for terms in _ACTION_EFFECT_TERMS.values()),
)
_READ_ONLY_ACTIONS = frozenset({"observe", "noop", "step"})


class StateBundleInference:
    """Observable-only StateBundle inference service."""

    def __init__(
        self,
        model: nn.Module,
        config: StateBundleInferenceConfig,
        *,
        token_estimator: TokenEstimator | None = None,
        neighbor_index_factory: Any | None = None,
    ):
        if not isinstance(model, nn.Module):
            raise TypeError("model must be a torch module")
        self.model = model
        self.config = config
        # Exact compact serialization is the production and public default so
        # budget diagnostics mean what they say.  CharacterTokenEstimator
        # remains available only as an explicitly selected fallback.
        self.token_estimator = token_estimator or CompactAgentTokenEstimator()
        self.neighbor_index_factory = neighbor_index_factory
        # Standalone callers get one active bounded memory.  The production
        # processor explicitly brackets each benchmark episode and keeps setup
        # cuts inactive until the first observation delivered to the agent.
        self._memory = _EpisodeMemory(active=True)

    def begin_episode(self) -> None:
        """Reset inference memory while setup-only transforms are still inactive."""

        self._memory = _EpisodeMemory(active=False)

    def activate_episode(self) -> None:
        """Start a clean agent-visible episode after environment setup."""

        self._memory = _EpisodeMemory(active=True)

    def end_episode(self) -> None:
        """Release all retained evidence when the task terminates."""

        self._memory = _EpisodeMemory(active=False)

    @staticmethod
    def _copy_memory(memory: _EpisodeMemory) -> _EpisodeMemory:
        retained = {
            key: _RetainedEvidence(
                memory_key=value.memory_key,
                observation=value.observation,
                aspect=value.aspect.detach().clone(),
                learned_score=value.learned_score,
                signals=value.signals,
                estimated_cost=value.estimated_cost,
                components=value.components,
                final_score=value.final_score,
                admitted_cut=value.admitted_cut,
                expires_cut=value.expires_cut,
                unresolved_kind=value.unresolved_kind,
                pre_action=value.pre_action,
            )
            for key, value in memory.retained.items()
        }
        return _EpisodeMemory(
            active=memory.active,
            cut_index=memory.cut_index,
            retained=retained,
            resolution_streaks=dict(memory.resolution_streaks),
            stable_resolution_tombstones=dict(memory.stable_resolution_tombstones),
            last_selected_logical_keys=set(memory.last_selected_logical_keys),
            actions=[dict(item) for item in memory.actions],
            evaluation_actions=[dict(item) for item in memory.evaluation_actions],
            next_action_sequence=memory.next_action_sequence,
            causal_cut_key=memory.causal_cut_key,
            processed_action_fingerprints=set(memory.processed_action_fingerprints),
            cache_key=memory.cache_key,
            cached_output=memory.cached_output,
        )

    @staticmethod
    def _safe_action(action: Mapping[str, Any] | None) -> dict[str, Any] | None:
        if action is None:
            return None
        if not isinstance(action, Mapping):
            raise TypeError("action context must be an object")
        action_type = action.get("action_type")
        if not isinstance(action_type, str) or not action_type.strip():
            raise ValueError("action context requires a non-empty action_type")
        parameters = action.get("parameters", {})
        if not isinstance(parameters, Mapping):
            raise TypeError("action context parameters must be an object")
        forbidden = _FORBIDDEN_ACTION_CONTEXT_KEYS | {
            "active_fault",
            "active_faults",
            "evaluator",
            "expected",
            "fault_type",
            "score_hints",
            "success_criteria",
        }

        def copy_safe(value: Any, path: str) -> Any:
            if isinstance(value, Mapping):
                copied: dict[str, Any] = {}
                for raw_key, item in value.items():
                    key = str(raw_key)
                    normalized_key = re.sub(
                        r"[^a-z0-9]+", "_", key.strip().lower()
                    ).strip("_")
                    if normalized_key in forbidden:
                        raise ValueError(
                            f"non-inference action field is not allowed at {path}.{key}"
                        )
                    copied[key] = copy_safe(item, f"{path}.{key}")
                return copied
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                return [copy_safe(item, f"{path}[]") for item in value]
            if value is None or isinstance(value, (str, bool, int, float)):
                if isinstance(value, float) and not math.isfinite(value):
                    raise ValueError(f"non-finite action value at {path}")
                return value
            raise TypeError(f"unsupported action value at {path}")

        return {
            "action_type": action_type.strip(),
            "parameters": copy_safe(parameters, "action.parameters"),
        }

    @staticmethod
    def _is_intervention(action: Mapping[str, Any] | None) -> bool:
        return bool(
            action
            and str(action.get("action_type", "")).lower() not in _READ_ONLY_ACTIONS
        )

    @staticmethod
    def _trace_is_action_evidence(
        observation: CanonicalObservation,
    ) -> bool:
        if observation.channel.value != "trace":
            return False
        payload = observation.payload
        status_counts = payload.get("status_counts", {})

        def status_is_healthy(status: Any) -> bool:
            normalized = str(status).strip().lower()
            if normalized in {"ok", "success", "successful", "2xx"}:
                return True
            if re.fullmatch(r"\d{3}", normalized):
                return 200 <= int(normalized) < 300
            return False

        unhealthy_status = isinstance(status_counts, Mapping) and any(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and float(value) > 0.0
            and not status_is_healthy(status)
            for status, value in status_counts.items()
        )
        retry_count = payload.get("retry_count", 0)
        retries = (
            isinstance(retry_count, (int, float))
            and not isinstance(retry_count, bool)
            and float(retry_count) > 0.0
        )
        critical_path = payload.get("critical_path") is True
        searchable = " ".join(
            str(payload.get(field, ""))
            for field in ("operation", "source", "destination")
        ).lower()
        operational_operation = any(
            term in searchable for term in _ACTION_EVIDENCE_TERMS
        )
        return unhealthy_status or retries or critical_path or operational_operation

    @classmethod
    def _is_potential_action_evidence(
        cls, candidate: _Candidate | _RetainedEvidence
    ) -> bool:
        """Keep a bounded pool that may be useful to a later intervention."""

        signal = candidate.signals
        if signal.active_alert or signal.slo_violation:
            return True
        if candidate.observation.channel.value == "alert":
            return True
        if cls._trace_is_action_evidence(candidate.observation):
            return True
        payload = candidate.observation.payload
        searchable = " ".join(
            (
                signal.logical_key,
                signal.metric_name or "",
                candidate.observation.metadata.primary_subsystem,
                str(payload.get("event_type", "")),
                str(payload.get("template", "")),
            )
        ).lower()
        return any(term in searchable for term in _KNOWN_ACTION_EVIDENCE_TERMS)

    @staticmethod
    def _action_scope(
        action: Mapping[str, Any],
    ) -> tuple[frozenset[str], frozenset[str], frozenset[str]]:
        action_type = str(action.get("action_type", "")).strip().lower()
        action_type_tokens = set(re.findall(r"[a-z0-9]+", action_type))
        parameters = action.get("parameters", {})
        parameter_text = json.dumps(
            parameters,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            default=str,
        ).lower()
        parameter_tokens = set(re.findall(r"[a-z0-9]+", parameter_text))

        entity_values: set[str] = set()

        def collect_strings(value: Any) -> None:
            if isinstance(value, Mapping):
                for key, item in value.items():
                    # Object keys can themselves be target IDs, for example
                    # load-balancer backend weight maps.
                    if isinstance(key, str) and any(
                        term in str(key).lower()
                        for term in ("rack", "server", "backend", "tenant", "unit")
                    ):
                        entity_values.add(str(key).strip().lower())
                    collect_strings(item)
            elif isinstance(value, Sequence) and not isinstance(
                value, (str, bytes, bytearray)
            ):
                for item in value:
                    collect_strings(item)
            elif isinstance(value, str) and value.strip():
                entity_values.add(value.strip().lower())

        collect_strings(parameters)
        inferred_domains = action_type_tokens - _ACTION_VERB_TERMS
        domains = set(_ACTION_DOMAIN_TERMS.get(action_type, frozenset()))
        domains.update(inferred_domains)
        effects = set(_ACTION_EFFECT_TERMS.get(action_type, frozenset()))
        effects.update(action_type_tokens)
        effects.update(parameter_tokens)
        return frozenset(domains), frozenset(effects), frozenset(entity_values)

    @classmethod
    def _action_relevant(
        cls,
        observation: CanonicalObservation,
        signal: ObservableSignals,
        action: Mapping[str, Any],
    ) -> bool:
        if signal.slo_violation:
            return True
        domains, effects, action_entities = cls._action_scope(action)
        candidate_entities = {
            value.strip().lower()
            for value in (*signal.entity_ids, *signal.target_ids)
            if value.strip()
        }
        if candidate_entities & action_entities:
            return True
        # Use semantic payload fields rather than the full schema envelope.
        # Generic keys such as ``latency_ms`` exist on every trace and must not
        # make an unrelated healthy operation relevant to a throttle action.
        semantic_payload = " ".join(
            str(observation.payload.get(field, ""))
            for field in (
                "metric_name",
                "alert_type",
                "message",
                "target",
                "event_type",
                "template",
                "operation",
                "source",
                "destination",
                "scope",
                "path",
            )
        )
        searchable = " ".join(
            (
                signal.metric_name or "",
                signal.alert_name or "",
                observation.metadata.primary_subsystem,
                semantic_payload,
            )
        ).lower()
        candidate_tokens = set(re.findall(r"[a-z0-9]+", searchable))
        specific_effects = (
            effects
            - domains
            - _ACTION_VERB_TERMS
            - {
                "target",
                "id",
                "c",
                "percent",
                "second",
                "seconds",
            }
        )
        specific_effects = {token for token in specific_effects if not token.isdigit()}
        return bool(candidate_tokens & domains) and bool(
            candidate_tokens & specific_effects
        )

    @classmethod
    def _is_action_evidence(
        cls,
        candidate: _Candidate | _RetainedEvidence,
        action: Mapping[str, Any] | None = None,
    ) -> bool:
        if action is None:
            return cls._is_potential_action_evidence(candidate)
        return cls._action_relevant(candidate.observation, candidate.signals, action)

    @staticmethod
    def _request_cache_key(request: AgentObservationRequest | None) -> tuple[Any, ...]:
        return (
            request.view_key
            if request is not None
            else AgentObservationRequest().view_key
        )

    def _cache_key(
        self,
        incident: CanonicalIncident,
        request: AgentObservationRequest | None,
        action: Mapping[str, Any] | None,
    ) -> tuple[Any, ...]:
        action_text = json.dumps(
            action,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        return (
            *self._causal_cut_key(incident),
            self._request_cache_key(request),
            action_text,
        )

    @staticmethod
    def _causal_cut_key(incident: CanonicalIncident) -> tuple[Any, ...]:
        """Identify simulator state independently of the requested view."""

        return (
            incident.incident_id,
            incident.snapshot_id,
            incident.query_time_seconds,
            incident.query_watermark_sequence,
            tuple(sorted(item.observation_id for item in incident.observations)),
        )

    @staticmethod
    def _clone_groups(groups: Sequence[_WorkingGroup]) -> list[_WorkingGroup]:
        return [_WorkingGroup(group.anchor, list(group.supports)) for group in groups]

    @staticmethod
    def _target_scope_summary(
        groups: Sequence[_WorkingGroup],
    ) -> tuple[bool, tuple[dict[str, Any], ...]]:
        """Summarize selector-estimated scopes represented in a trial bundle.

        Only direct and downstream abnormal scopes are candidates for target
        ambiguity. Healthy contextual peers remain available in the telemetry
        tables but are intentionally not advertised as possible targets.
        """

        role_priority = {
            "direct_target_candidate": 0,
            "downstream_affected_scope": 1,
            "contextual_peer": 2,
        }
        by_scope: dict[str, list[_Candidate]] = defaultdict(list)
        for group in groups:
            for candidate in (group.anchor, *group.supports):
                role = candidate.signals.estimated_target_role
                if role == "contextual_peer":
                    continue
                for scope in candidate.signals.estimated_scope_ids:
                    if scope:
                        by_scope[scope].append(candidate)
        summary: list[dict[str, Any]] = []
        for scope, members in by_scope.items():
            estimated_role = min(
                (item.signals.estimated_target_role for item in members),
                key=lambda role: (role_priority.get(role, 3), role),
            )
            summary.append(
                {
                    "scope": scope,
                    "estimated_role": estimated_role,
                    "supporting_observation_ids": sorted(
                        {item.identifier for item in members}
                    ),
                }
            )
        summary.sort(
            key=lambda item: (
                role_priority.get(str(item["estimated_role"]), 3),
                str(item["scope"]),
            )
        )
        return len(summary) >= 2, tuple(summary)

    def _bundle_cost(
        self,
        groups: Sequence[_WorkingGroup],
        *,
        query_time_seconds: float,
        request: AgentObservationRequest | None,
    ) -> int:
        specifications = tuple(
            (
                group.anchor.observation,
                tuple(item.observation for item in group.supports),
            )
            for group in groups
        )
        estimator = getattr(self.token_estimator, "estimate_bundle", None)
        if callable(estimator):
            ambiguity, target_scopes = self._target_scope_summary(groups)
            extra: dict[str, Any] = {}
            if isinstance(self.token_estimator, CompactAgentTokenEstimator):
                extra = {
                    "target_scope_ambiguity": ambiguity,
                    "target_scope_candidates": target_scopes,
                }
            return int(
                estimator(
                    specifications,
                    self.config.redaction_policy,
                    query_time_seconds=query_time_seconds,
                    request=request,
                    **extra,
                )
            )
        seen: set[str] = set()
        total = 0
        for anchor, supports in specifications:
            for observation in (anchor, *supports):
                if observation.observation_id in seen:
                    continue
                seen.add(observation.observation_id)
                total += self.token_estimator.estimate(
                    observation, self.config.redaction_policy
                )
        return total

    def _trial_with_anchor(
        self,
        groups: Sequence[_WorkingGroup],
        candidate: _Candidate,
    ) -> list[_WorkingGroup]:
        trial = self._clone_groups(groups)
        trial.append(_WorkingGroup(candidate))
        return trial

    def _trial_with_support(
        self,
        groups: Sequence[_WorkingGroup],
        group_index: int,
        candidate: _Candidate,
    ) -> list[_WorkingGroup]:
        trial = self._clone_groups(groups)
        trial[group_index].supports.append(candidate)
        return trial

    def _ordered_support_trials(
        self,
        groups: Sequence[_WorkingGroup],
        candidate: _Candidate,
    ) -> list[list[_WorkingGroup]]:
        placements = sorted(
            range(len(groups)),
            key=lambda group_index: (
                -self._support_value(groups[group_index].anchor, candidate),
                groups[group_index].anchor.identifier,
            ),
        )
        return [
            self._trial_with_support(groups, group_index, candidate)
            for group_index in placements
        ]

    @staticmethod
    def _protected_support_related(
        anchor: _Candidate,
        candidate: _Candidate,
    ) -> bool:
        """Require an observable relationship before naming protected support.

        This is deliberately independent of evaluator labels. It prevents the
        exact-cost packer from representing unrelated protected facts as
        corroboration solely because one JSON group envelope is cheaper.
        """

        generic_entities = {
            "application",
            "datacenter",
            "global",
            "monitoring",
            "network",
            "storage",
            "unknown",
            "workload",
        }
        shared_entities = {
            value.lower()
            for value in set(anchor.signals.entity_ids)
            & set(candidate.signals.entity_ids)
            if value.lower() not in generic_entities
        }
        shared_correlations = bool(
            set(anchor.observation.metadata.correlation_ids.items())
            & set(candidate.observation.metadata.correlation_ids.items())
        )
        same_subsystem = (
            anchor.observation.metadata.primary_subsystem
            == candidate.observation.metadata.primary_subsystem
            and anchor.observation.metadata.primary_subsystem.lower()
            not in {"unknown", "datacenter"}
        )
        same_alert = bool(
            anchor.signals.alert_name
            and anchor.signals.alert_name == candidate.signals.alert_name
        )
        same_metric = bool(
            anchor.signals.metric_name
            and anchor.signals.metric_name == candidate.signals.metric_name
        )
        same_targeted_request = (
            anchor.signals.request_match and candidate.signals.request_match
        )
        alert_metric_link = (
            anchor.signals.active_alert
            and candidate.signals.linked_to_active_alert
            or candidate.signals.active_alert
            and anchor.signals.linked_to_active_alert
        ) and bool(
            set((*anchor.signals.entity_ids, *anchor.signals.target_ids))
            & set((*candidate.signals.entity_ids, *candidate.signals.target_ids))
        )
        return bool(
            shared_entities
            or shared_correlations
            or same_subsystem
            or same_alert
            or same_metric
            or same_targeted_request
            or alert_metric_link
        )

    def _ordered_protected_support_trials(
        self,
        groups: Sequence[_WorkingGroup],
        candidate: _Candidate,
    ) -> list[list[_WorkingGroup]]:
        placements = sorted(
            (
                group_index
                for group_index, group in enumerate(groups)
                if self._protected_support_related(group.anchor, candidate)
            ),
            key=lambda group_index: (
                -self._support_value(groups[group_index].anchor, candidate),
                groups[group_index].anchor.identifier,
            ),
        )
        return [
            self._trial_with_support(groups, group_index, candidate)
            for group_index in placements
        ]

    @staticmethod
    def _without_candidate(
        groups: Sequence[_WorkingGroup], identifier: str
    ) -> list[_WorkingGroup]:
        """Remove one row while preserving every other selected observation."""

        retained: list[_WorkingGroup] = []
        for group in groups:
            supports = [
                item for item in group.supports if item.identifier != identifier
            ]
            if group.anchor.identifier != identifier:
                retained.append(_WorkingGroup(group.anchor, supports))
                continue
            if supports:
                retained.append(_WorkingGroup(supports[0], supports[1:]))
        return retained

    def _fits(
        self,
        groups: Sequence[_WorkingGroup],
        *,
        query_time_seconds: float,
        request: AgentObservationRequest | None,
    ) -> bool:
        return (
            self._bundle_cost(
                groups,
                query_time_seconds=query_time_seconds,
                request=request,
            )
            <= self.config.total_token_budget
        )

    @staticmethod
    def _severity_priority(candidate: _Candidate) -> int:
        return {
            "critical": 4,
            "high": 3,
            "warning": 2,
            "info": 1,
        }.get(candidate.signals.severity or "", 0)

    def _protected_rank(self, candidate: _Candidate) -> tuple[Any, ...]:
        linkage_priority = {
            "direct_counterpart": 3,
            "entity_and_symptom_match": 2,
            "weak_context_match": 1,
            "no_match": 0,
        }.get(candidate.signals.alert_linkage_strength, 0)
        return (
            -int(candidate.signals.request_match),
            -int(
                candidate.signals.active_alert
                and candidate.signals.severity in {"critical", "high"}
            ),
            -int(candidate.sticky),
            -linkage_priority,
            -self._severity_priority(candidate),
            -int(candidate.signals.target_bearing),
            -int(candidate.signals.slo_violation),
            -int(
                candidate.signals.estimated_target_role
                == "direct_target_candidate"
            ),
            -int(candidate.signals.anomalous_metric),
            -candidate.observation.metadata.event_end_time_seconds,
            -candidate.final_score,
            candidate.identifier,
        )

    def _support_value(self, anchor: _Candidate, candidate: _Candidate) -> float:
        return _support_score(
            anchor.observation,
            candidate.observation,
            anchor.aspect,
            candidate.aspect,
            candidate.learned_score,
            self.config,
        ) + sum(value for _, value in candidate.components)

    def _candidate_diagnostic(self, candidate: _Candidate) -> dict[str, Any]:
        return {
            "observation_id": candidate.identifier,
            "channel": candidate.observation.channel.value,
            "logical_key": candidate.signals.logical_key,
            "learned_relevance_score": candidate.learned_score,
            "learned_rank": candidate.learned_rank,
            "learned_top_m": candidate.learned_top_m,
            "learned_score_weight": self.config.learned_score_weight,
            "reranking_contributions": {
                name: value for name, value in candidate.components
            },
            "final_ranking_score": candidate.final_score,
            "final_rank": candidate.final_rank,
            "protected": candidate.protected,
            "protection_reasons": list(candidate.protected_reasons),
            "alert_linkage_strength": candidate.signals.alert_linkage_strength,
            "alert_linkage_reason": candidate.signals.alert_linkage_reason,
            "matched_alert_field": candidate.signals.matched_alert_field,
            "matched_alert_entity": candidate.signals.matched_alert_entity,
            "matched_alert_observation_id": (
                candidate.signals.matched_alert_observation_id
            ),
            "equivalence_group_id": candidate.signals.equivalence_group_id,
            "equivalence_representative_id": (
                candidate.equivalence_representative_id
            ),
            "equivalence_rank": candidate.equivalence_rank,
            "estimated_target_role": candidate.signals.estimated_target_role,
            "estimated_scope_ids": list(candidate.signals.estimated_scope_ids),
            "sticky": candidate.sticky,
            "retained": candidate.retained,
            "post_action": candidate.post_action,
            "semantic_redundant": candidate.redundant,
            "selected_as": candidate.selected_as or "rejected",
            "selection_phase": candidate.selection_phase,
            "rejection_reason": candidate.rejection_reason,
            "eviction_reason": candidate.eviction_reason,
            "estimated_token_cost": candidate.estimated_cost,
            "remaining_budget_at_decision": candidate.remaining_budget_at_decision,
            "observable_features": {
                "entity_ids": list(candidate.signals.entity_ids),
                "target_ids": list(candidate.signals.target_ids),
                "active_alert": candidate.signals.active_alert,
                "severity": candidate.signals.severity,
                "target_bearing": candidate.signals.target_bearing,
                "slo_violation": candidate.signals.slo_violation,
                "linked_to_active_alert": candidate.signals.linked_to_active_alert,
                "anomalous_metric": candidate.signals.anomalous_metric,
                "zero_series": candidate.signals.zero_series,
                "constant_series": candidate.signals.constant_series,
                "normal_comparison": candidate.signals.normal_comparison,
                "recent_config_change": candidate.signals.recent_config_change,
                "static_config": candidate.signals.static_config,
                "stale_config": candidate.signals.stale_config,
                "bookkeeping_log": candidate.signals.bookkeeping_log,
                "action_relevant_trace": self._trace_is_action_evidence(
                    candidate.observation
                ),
                "request_eligible": candidate.signals.request_eligible,
                "request_match": candidate.signals.request_match,
            },
        }

    @torch.no_grad()
    def infer(
        self,
        incident: CanonicalIncident,
        *,
        request: AgentObservationRequest | None = None,
        action: Mapping[str, Any] | None = None,
    ) -> StateBundleOutput:
        """Construct a guarded bundle from causal, inference-visible evidence.

        The model forward pass and its learned scores/aspects are unchanged.
        ``request`` and ``action`` contain only controls issued by the agent;
        simulator/evaluator annotations are neither accepted nor consulted.
        """

        if not isinstance(incident, CanonicalIncident):
            raise TypeError(
                "infer accepts a CanonicalIncident and no training annotations"
            )
        if request is not None and not isinstance(request, AgentObservationRequest):
            raise TypeError("request must be an AgentObservationRequest or None")
        safe_action = self._safe_action(action)
        for observation in incident.observations:
            observation.validate_available(
                incident.query_time_seconds, incident.query_watermark_sequence
            )

        cache_key = self._cache_key(incident, request, safe_action)
        if (
            self._memory.active
            and self._memory.cache_key == cache_key
            and self._memory.cached_output is not None
        ):
            return self._memory.cached_output

        was_training = self.model.training
        self.model.eval()
        try:
            output = cast(
                _InferenceModelOutput,
                self.model(IncidentBatch((incident,))),
            )
        finally:
            self.model.train(was_training)

        count = len(incident.observations)
        aspects = F.normalize(output.aspect_embeddings[0, :count], dim=-1)
        relevance = output.relevance_scores[0, :count]
        signals = analyze_observations(
            incident.observations,
            query_time_seconds=incident.query_time_seconds,
            request=request,
            config=self.config,
        )
        memory = (
            self._copy_memory(self._memory)
            if self._memory.active
            else _EpisodeMemory(active=False)
        )
        if (
            memory.active
            and memory.causal_cut_key is not None
            and memory.causal_cut_key[0] != incident.incident_id
        ):
            # Standalone callers do not necessarily bracket episodes through
            # the runtime processor.  Never carry operational evidence across
            # incident identities.
            memory = _EpisodeMemory(active=True)
        if (
            memory.active
            and memory.causal_cut_key is not None
            and incident.incident_id == memory.causal_cut_key[0]
            and incident.query_time_seconds < float(memory.causal_cut_key[2])
        ):
            raise ValueError("causal cuts must not move backwards within an episode")
        if (
            memory.active
            and memory.causal_cut_key is not None
            and incident.incident_id == memory.causal_cut_key[0]
            and isinstance(memory.causal_cut_key[3], int)
            and not isinstance(memory.causal_cut_key[3], bool)
            and isinstance(incident.query_watermark_sequence, int)
            and not isinstance(incident.query_watermark_sequence, bool)
            and incident.query_watermark_sequence < memory.causal_cut_key[3]
        ):
            raise ValueError(
                "causal watermark must not move backwards within an episode"
            )
        causal_cut_key = self._causal_cut_key(incident)
        causal_cut_advanced = memory.causal_cut_key != causal_cut_key
        cut_index = memory.cut_index + int(causal_cut_advanced)
        if cut_index < 1:
            cut_index = 1
        if causal_cut_advanced:
            # Action fingerprints only deduplicate repeated transformations of
            # one simulator state.  A later causal cut may legitimately repeat
            # the same operational action.
            memory.processed_action_fingerprints.clear()
        memory_evictions: list[dict[str, Any]] = []

        # An intervention opens a compact pre/post comparison epoch.  Only
        # previously selected operational evidence can enter this memory.
        action_fingerprint = json.dumps(
            safe_action,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        if (
            self._is_intervention(safe_action)
            and action_fingerprint not in memory.processed_action_fingerprints
        ):
            action_number = memory.next_action_sequence
            memory.next_action_sequence += 1
            for key, entry in tuple(memory.retained.items()):
                if entry.pre_action or not self._is_action_evidence(entry, safe_action):
                    continue
                pre_key = f"pre-action-{action_number}:{key}"
                memory.retained[pre_key] = _RetainedEvidence(
                    memory_key=pre_key,
                    observation=entry.observation,
                    aspect=entry.aspect.detach().clone(),
                    learned_score=entry.learned_score,
                    signals=entry.signals,
                    estimated_cost=entry.estimated_cost,
                    components=entry.components,
                    final_score=entry.final_score,
                    admitted_cut=cut_index,
                    expires_cut=cut_index + self.config.post_action_retention_cuts,
                    unresolved_kind=entry.unresolved_kind,
                    pre_action=True,
                )
            action_record = {
                "sequence": action_number,
                "cut_index": cut_index,
                "expires_cut": cut_index + self.config.post_action_retention_cuts,
                **(safe_action or {}),
            }
            memory.actions.append(action_record)
            memory.actions = memory.actions[-self.config.retention_max_cuts :]
            memory.evaluation_actions.append(dict(action_record))
            memory.processed_action_fingerprints.add(action_fingerprint)

        # Action-effect windows have their own lifetime.  Do not truncate an
        # unexpired action merely because ordinary evidence retention uses a
        # smaller cut cap; discard the compact action record once its effect
        # window has closed instead.
        memory.evaluation_actions = [
            item
            for item in memory.evaluation_actions
            if cut_index
            <= int(
                item.get(
                    "expires_cut",
                    int(item.get("cut_index", cut_index))
                    + self.config.post_action_retention_cuts,
                )
            )
        ]

        current_by_logical: dict[str, list[int]] = defaultdict(list)
        for index, signal in enumerate(signals):
            current_by_logical[signal.logical_key].append(index)
        # A genuinely recurring condition is eligible for retention again.
        # Otherwise a stable-resolution tombstone prevents alternate views of
        # this or later cuts from resurrecting the released normal row.
        for signal in signals:
            if signal.active_alert or signal.slo_violation:
                memory.stable_resolution_tombstones.pop(signal.logical_key, None)

        sticky_keys: set[str] = set()
        sticky_reasons: dict[str, str] = {}
        for memory_key, entry in tuple(memory.retained.items()):
            if entry.pre_action:
                if cut_index > entry.expires_cut:
                    memory.retained.pop(memory_key, None)
                    memory.resolution_streaks.pop(memory_key, None)
                    memory_evictions.append(
                        {
                            "logical_key": entry.signals.logical_key,
                            "observation_id": entry.observation.observation_id,
                            "reason": "post_action_effect_evaluated",
                        }
                    )
                continue
            current_indices = current_by_logical.get(entry.signals.logical_key, [])
            if current_indices:
                condition_active = any(
                    signals[index].active_alert or signals[index].slo_violation
                    for index in current_indices
                )
                if entry.unresolved_kind and not condition_active:
                    streak = memory.resolution_streaks.get(memory_key, 0)
                    if causal_cut_advanced:
                        streak += 1
                        memory.resolution_streaks[memory_key] = streak
                    if streak >= self.config.resolution_stability_cuts:
                        memory.retained.pop(memory_key, None)
                        memory.resolution_streaks.pop(memory_key, None)
                        memory.stable_resolution_tombstones[
                            entry.signals.logical_key
                        ] = cut_index
                        memory_evictions.append(
                            {
                                "logical_key": entry.signals.logical_key,
                                "observation_id": entry.observation.observation_id,
                                "reason": "condition_resolved_stably",
                            }
                        )
                        continue
                elif causal_cut_advanced:
                    memory.resolution_streaks[memory_key] = 0
                if entry.unresolved_kind or cut_index <= entry.expires_cut:
                    sticky_keys.add(entry.signals.logical_key)
                    sticky_reasons[entry.signals.logical_key] = (
                        f"retained_unresolved_{entry.unresolved_kind}"
                        if entry.unresolved_kind
                        else "retained_important_anchor"
                    )
                else:
                    memory.retained.pop(memory_key, None)
                    memory.resolution_streaks.pop(memory_key, None)
                    memory_evictions.append(
                        {
                            "logical_key": entry.signals.logical_key,
                            "observation_id": entry.observation.observation_id,
                            "reason": "retention_window_expired",
                        }
                    )
            elif entry.unresolved_kind:
                # Absence from a complete canonical cut is treated as a
                # tentative resolution, not as permission to retain a stale
                # firing condition forever.  Require the same stability streak
                # as an explicit resolved row to tolerate transient telemetry
                # gaps before releasing it.
                streak = memory.resolution_streaks.get(memory_key, 0)
                if causal_cut_advanced:
                    streak += 1
                    memory.resolution_streaks[memory_key] = streak
                if streak >= self.config.resolution_stability_cuts:
                    memory.retained.pop(memory_key, None)
                    memory.resolution_streaks.pop(memory_key, None)
                    memory.stable_resolution_tombstones[entry.signals.logical_key] = (
                        cut_index
                    )
                    memory_evictions.append(
                        {
                            "logical_key": entry.signals.logical_key,
                            "observation_id": entry.observation.observation_id,
                            "reason": "condition_absent_stably",
                        }
                    )
            elif cut_index > entry.expires_cut:
                memory.retained.pop(memory_key, None)
                memory.resolution_streaks.pop(memory_key, None)
                memory_evictions.append(
                    {
                        "logical_key": entry.signals.logical_key,
                        "observation_id": entry.observation.observation_id,
                        "reason": "retention_window_expired",
                    }
                )

        # Mitigations can overlap.  Preserve each action's own evaluation
        # horizon so a newer intervention cannot hide delayed evidence for an
        # older, still-unresolved one.
        evaluation_actions: tuple[Mapping[str, Any], ...] = tuple(
            item
            for item in memory.evaluation_actions
            if cut_index
            <= int(
                item.get(
                    "expires_cut",
                    int(item.get("cut_index", cut_index))
                    + self.config.post_action_retention_cuts,
                )
            )
        )
        post_action_active = bool(evaluation_actions)
        all_current: list[_Candidate] = []
        for index, (observation, signal) in enumerate(
            zip(incident.observations, signals)
        ):
            sticky = signal.logical_key in sticky_keys
            action_relevant = any(
                self._action_relevant(observation, signal, evaluation_action)
                for evaluation_action in evaluation_actions
            )
            post_action = post_action_active and (
                signal.slo_violation
                or (
                    action_relevant
                    # A fleet of unchanged zero-valued health counters must
                    # not become hard-protected merely because an action was
                    # taken. Explicit linkage or a targeted request can still
                    # preserve an individually useful zero comparison.
                    and (
                        not signal.zero_series
                        or signal.linked_to_active_alert
                        or signal.request_match
                    )
                )
            )
            protection = list(signal.protection_reasons)
            if sticky:
                protection.append(sticky_reasons[signal.logical_key])
            if post_action:
                protection.append("post_action_evaluation")
            components = reranking_components(
                signal,
                self.config,
                sticky=sticky,
                post_action=post_action,
                redundant=False,
            )
            learned_score = float(relevance[index].item())
            all_current.append(
                _Candidate(
                    observation=observation,
                    observation_index=index,
                    aspect=aspects[index],
                    learned_score=learned_score,
                    signals=signal,
                    estimated_cost=self.token_estimator.estimate(
                        observation, self.config.redaction_policy
                    ),
                    sticky=sticky,
                    post_action=post_action,
                    protected_reasons=tuple(sorted(set(protection))),
                    components=components,
                    final_score=(
                        self.config.learned_score_weight * learned_score
                        + sum(value for _, value in components)
                    ),
                )
            )

        # Retained observations are added only when the current cut does not
        # supersede the same fact, or when explicitly preserved as pre-action.
        prospective_ids = {item.identifier for item in all_current}
        retained_rows: list[
            tuple[str, _RetainedEvidence, CanonicalObservation, int]
        ] = []
        synthetic_index = count
        for memory_key, entry in sorted(memory.retained.items()):
            if not entry.pre_action and entry.signals.logical_key in current_by_logical:
                continue
            retained_observation = entry.observation
            if entry.pre_action:
                # Aggregate canonical rows often keep a stable observation ID
                # while their values evolve.  A deterministic retained ID lets
                # the bundle represent both sides of the comparison without
                # mutating the saved canonical observation or conflating it
                # with the current row.
                retained_observation = replace(
                    entry.observation,
                    observation_id=(
                        f"retained:{memory_key}:{entry.observation.observation_id}"
                    ),
                )
            if retained_observation.observation_id in prospective_ids:
                if not entry.pre_action:
                    memory.retained.pop(memory_key, None)
                    memory.resolution_streaks.pop(memory_key, None)
                    memory_evictions.append(
                        {
                            "logical_key": entry.signals.logical_key,
                            "observation_id": entry.observation.observation_id,
                            "reason": "duplicate_superseded_by_current",
                        }
                    )
                continue
            if entry.pre_action and cut_index > entry.expires_cut:
                continue
            retained_rows.append(
                (memory_key, entry, retained_observation, synthetic_index)
            )
            prospective_ids.add(retained_observation.observation_id)
            synthetic_index += 1

        # Request scope (especially log_limit) is a set-wise property. Analyze
        # retained rows jointly with the current canonical corpus so each view
        # has exactly one deterministic eligibility decision.
        combined_scope_signals = analyze_observations(
            (
                *incident.observations,
                *(row[2] for row in retained_rows),
            ),
            query_time_seconds=incident.query_time_seconds,
            request=request,
            config=self.config,
        )
        for index, candidate in enumerate(all_current):
            scope_signal = combined_scope_signals[index]
            if (
                candidate.signals.request_eligible == scope_signal.request_eligible
                and candidate.signals.request_match == scope_signal.request_match
            ):
                continue
            candidate.signals = replace(
                candidate.signals,
                request_eligible=scope_signal.request_eligible,
                request_match=scope_signal.request_match,
            )
            protection = [
                reason
                for reason in candidate.protected_reasons
                if reason != "targeted_request"
            ]
            if candidate.signals.request_match:
                protection.append("targeted_request")
            candidate.protected_reasons = tuple(sorted(set(protection)))
            candidate.components = reranking_components(
                candidate.signals,
                self.config,
                sticky=candidate.sticky,
                post_action=candidate.post_action,
                redundant=candidate.redundant,
            )
            candidate.final_score = (
                self.config.learned_score_weight * candidate.learned_score
                + sum(value for _, value in candidate.components)
            )
            if not candidate.signals.request_eligible:
                candidate.rejection_reason = "outside_request_scope"

        retained_scope_by_id = {
            row[2].observation_id: combined_scope_signals[count + offset]
            for offset, row in enumerate(retained_rows)
        }
        all_retained: list[_Candidate] = []
        for memory_key, entry, retained_observation, observation_index in retained_rows:
            current_request_signal = retained_scope_by_id[
                retained_observation.observation_id
            ]
            retained_signal = replace(
                entry.signals,
                request_eligible=current_request_signal.request_eligible,
                request_match=current_request_signal.request_match,
            )
            protection = [
                (
                    "retained_pre_action_evidence"
                    if entry.pre_action
                    else (
                        f"retained_unresolved_{entry.unresolved_kind}"
                        if entry.unresolved_kind
                        else "retained_important_anchor"
                    )
                )
            ]
            if retained_signal.request_match:
                protection.append("targeted_request")
            components = reranking_components(
                retained_signal,
                self.config,
                sticky=True,
                post_action=entry.pre_action,
                redundant=False,
            )
            retained_candidate = _Candidate(
                observation=retained_observation,
                observation_index=observation_index,
                aspect=F.normalize(entry.aspect.to(aspects.device), dim=-1),
                learned_score=entry.learned_score,
                signals=retained_signal,
                estimated_cost=self.token_estimator.estimate(
                    retained_observation, self.config.redaction_policy
                ),
                sticky=True,
                retained=True,
                post_action=entry.pre_action,
                protected_reasons=tuple(protection),
                components=components,
                final_score=(
                    self.config.learned_score_weight * entry.learned_score
                    + sum(value for _, value in components)
                ),
            )
            all_retained.append(retained_candidate)
            if not retained_signal.request_eligible:
                retained_candidate.rejection_reason = "outside_request_scope"

        # Record the learned ordering before any deterministic policy
        # arbitration. Retained rows use their saved learned score and are
        # explicitly marked as such elsewhere; no score is recomputed or
        # altered here.
        learned_ranked_current = sorted(
            all_current,
            key=lambda item: (-item.learned_score, item.identifier),
        )
        for rank, candidate in enumerate(learned_ranked_current, start=1):
            candidate.learned_rank = rank
            candidate.learned_top_m = rank <= self.config.top_m
        for rank, candidate in enumerate(
            sorted(
                all_retained,
                key=lambda item: (-item.learned_score, item.identifier),
            ),
            start=len(learned_ranked_current) + 1,
        ):
            candidate.learned_rank = rank
            candidate.learned_top_m = False
        learned_top_m = {
            item.identifier
            for item in learned_ranked_current[: self.config.top_m]
        }

        preselection_protected_omissions: list[dict[str, Any]] = []
        # Finalize request scope before duplicate handling. This matters for
        # set-wise filters such as log_limit: a retained row can displace one
        # current log and thereby make a different current row the valid one.
        # Exact deduplication is global across current and ordinary retained
        # evidence. Synthetic pre-action rows are deliberately exempt because
        # they exist to preserve a before/after comparison.
        exact_groups: dict[tuple[Any, ...], list[_Candidate]] = defaultdict(list)
        for candidate in (*all_current, *all_retained):
            if not candidate.signals.request_eligible:
                candidate.rejection_reason = "outside_request_scope"
                continue
            candidate.rejection_reason = None
            signature = (
                ("pre_action", candidate.identifier)
                if candidate.retained and candidate.post_action
                else _duplicate_signature(
                    candidate.observation, self.config.redaction_policy
                )
            )
            exact_groups[signature].append(candidate)

        candidates = []
        for group_candidates in exact_groups.values():
            ordered = sorted(
                group_candidates,
                key=lambda item: (
                    # Prefer the current canonical representation to an
                    # equivalent ordinary memory row. Among memory rows keep
                    # the newest operational fact rather than resurrecting a
                    # stale high-scoring duplicate.
                    int(item.retained),
                    -item.observation.metadata.event_end_time_seconds,
                    -int(item.protected),
                    -item.final_score,
                    -item.learned_score,
                    item.identifier,
                ),
            )
            candidates.append(ordered[0])
            exact_representative = ordered[0]
            exact_representative.equivalence_representative_id = (
                exact_representative.identifier
            )
            exact_representative.equivalence_rank = 1
            for exact_rank, loser in enumerate(ordered[1:], start=2):
                loser.rejection_reason = "exact_duplicate"
                loser.hard_excluded = True
                loser.redundant = True
                loser.components = reranking_components(
                    loser.signals,
                    self.config,
                    sticky=loser.sticky,
                    post_action=loser.post_action,
                    redundant=True,
                )
                loser.final_score = (
                    self.config.learned_score_weight * loser.learned_score
                    + sum(value for _, value in loser.components)
                )
                loser.equivalence_representative_id = exact_representative.identifier
                loser.equivalence_rank = exact_rank
                if loser.protected:
                    preselection_protected_omissions.append(
                        {
                            "observation_id": loser.identifier,
                            "protection_reasons": list(loser.protected_reasons),
                            "protection_reason": loser.protected_reasons[0],
                            "equivalence_group_id": (
                                loser.signals.equivalence_group_id
                            ),
                            "reason": "protected_exact_duplicate",
                            "rank": exact_rank,
                            "estimated_token_cost": loser.estimated_cost,
                            "selected_representative": (
                                exact_representative.identifier
                            ),
                        }
                    )
                if loser.retained and not loser.post_action:
                    for retained_key, entry in tuple(memory.retained.items()):
                        if (
                            not entry.pre_action
                            and entry.observation.observation_id == loser.identifier
                        ):
                            memory.retained.pop(retained_key, None)
                            memory.resolution_streaks.pop(retained_key, None)
                            memory_evictions.append(
                                {
                                    "logical_key": entry.signals.logical_key,
                                    "observation_id": entry.observation.observation_id,
                                    "reason": "duplicate_superseded",
                                }
                            )

        # Deterministically arbitrate equivalent operational facts before
        # protected admission. This is a hard eligibility decision: excluded
        # members cannot return through anchors, supports, coverage, or fill.
        semantic_groups: dict[str, list[_Candidate]] = defaultdict(list)
        for candidate in candidates:
            semantic_key = (
                f"pre_action:{candidate.identifier}"
                if candidate.retained and candidate.post_action
                else candidate.signals.equivalence_group_id
            )
            semantic_groups[semantic_key].append(candidate)
        for group_candidates in semantic_groups.values():
            contextual_group = all(
                item.signals.estimated_target_role == "contextual_peer"
                for item in group_candidates
            )
            if any(item.signals.zero_series for item in group_candidates):
                allowed = min(
                    self.config.zero_series_representatives,
                    self.config.healthy_peer_representatives,
                )
            elif contextual_group:
                allowed = self.config.healthy_peer_representatives
            else:
                allowed = self.config.semantic_group_representatives
            ordered = sorted(
                group_candidates,
                key=lambda item: (
                    -int(item.signals.request_match),
                    -{
                        "direct_counterpart": 3,
                        "entity_and_symptom_match": 2,
                        "weak_context_match": 1,
                        "no_match": 0,
                    }.get(item.signals.alert_linkage_strength, 0),
                    -int(
                        item.signals.estimated_target_role
                        == "direct_target_candidate"
                    ),
                    -self._severity_priority(item),
                    -int(item.signals.slo_violation),
                    -int(item.signals.anomalous_metric),
                    -int(item.signals.target_bearing),
                    -item.observation.metadata.event_end_time_seconds,
                    -item.final_score,
                    item.identifier,
                ),
            )
            representative = ordered[0]
            for offset, candidate in enumerate(ordered, start=1):
                candidate.equivalence_rank = offset
                candidate.equivalence_representative_id = representative.identifier
                candidate.redundant = offset > allowed
                if candidate.redundant:
                    candidate.hard_excluded = True
                    candidate.rejection_reason = (
                        "protected_equivalence_cap"
                        if candidate.protected
                        else "equivalence_group_cap"
                    )
                candidate.components = reranking_components(
                    candidate.signals,
                    self.config,
                    sticky=candidate.sticky,
                    post_action=candidate.post_action,
                    redundant=candidate.redundant,
                )
                candidate.final_score = (
                    self.config.learned_score_weight * candidate.learned_score
                    + sum(value for _, value in candidate.components)
                )
                if candidate.redundant and candidate.protected:
                    preselection_protected_omissions.append(
                        {
                            "observation_id": candidate.identifier,
                            "protection_reasons": list(candidate.protected_reasons),
                            "protection_reason": candidate.protected_reasons[0],
                            "equivalence_group_id": (
                                candidate.signals.equivalence_group_id
                            ),
                            "reason": "protected_equivalence_cap",
                            "rank": offset,
                            "estimated_token_cost": candidate.estimated_cost,
                            "selected_representative": representative.identifier,
                        }
                    )

        final_ranked_all = sorted(
            (*all_current, *all_retained),
            key=lambda item: (-item.final_score, item.identifier),
        )
        for rank, candidate in enumerate(final_ranked_all, start=1):
            candidate.final_rank = rank

        unique_current = [item for item in candidates if not item.retained]

        reranked_order = sorted(
            unique_current,
            key=lambda item: (-item.final_score, item.identifier),
        )
        reranked_top_m = [
            item for item in reranked_order if not item.hard_excluded
        ][: self.config.top_m]
        targeted = request_is_targeted(request)
        request_matches = [item for item in candidates if item.signals.request_match]
        canonical_request_match_count = sum(
            item.signals.request_match for item in (*all_current, *all_retained)
        )
        constrain_to_matches = targeted and bool(request_matches)

        groups: list[_WorkingGroup] = []
        selected_ids: set[str] = set()
        protected_omissions: list[dict[str, Any]] = list(
            preselection_protected_omissions
        )
        protected_anchor_deferrals: list[dict[str, Any]] = []
        protected_repacks: list[dict[str, Any]] = []
        protected_selected: list[_Candidate] = []
        protected_order = sorted(
            (
                item
                for item in candidates
                if item.protected and not item.hard_excluded
            ),
            key=self._protected_rank,
        )
        for protected_rank, candidate in enumerate(protected_order, start=1):
            # Protected evidence bypasses ordinary count/score thresholds, but
            # it still shares the exact serialized budget. Consider all legal
            # placements and choose the cheapest fitting representation so
            # unnecessary group envelopes do not crowd out later evidence.
            anchor_trial = self._trial_with_anchor(groups, candidate)
            anchor_fits = self._fits(
                anchor_trial,
                query_time_seconds=incident.query_time_seconds,
                request=request,
            )
            placement_options: list[
                tuple[
                    int,
                    int,
                    str,
                    list[_WorkingGroup],
                    dict[str, Any] | None,
                ]
            ] = []
            if anchor_fits:
                placement_options.append(
                    (
                        self._bundle_cost(
                            anchor_trial,
                            query_time_seconds=incident.query_time_seconds,
                            request=request,
                        ),
                        0,
                        "anchor",
                        anchor_trial,
                        None,
                    )
                )
            for support_trial in self._ordered_protected_support_trials(
                groups, candidate
            ):
                if self._fits(
                    support_trial,
                    query_time_seconds=incident.query_time_seconds,
                    request=request,
                ):
                    placement_options.append(
                        (
                            self._bundle_cost(
                                support_trial,
                                query_time_seconds=incident.query_time_seconds,
                                request=request,
                            ),
                            1,
                            "support",
                            support_trial,
                            None,
                        )
                    )
            # A previously selected support can be the only observable bridge
            # to the new row. Try every member as a deterministic repack pivot,
            # both within its current group and across the complete protected
            # set. This avoids an omission caused solely by the first anchor's
            # identity while retaining the relation gate for every support.
            repack_trials: list[
                tuple[_Candidate, list[_WorkingGroup], tuple[_Candidate, ...]]
            ] = []
            for group_index, group in enumerate(groups):
                members = [group.anchor, *group.supports, candidate]
                for pivot in members:
                    if not all(
                        item.identifier == pivot.identifier
                        or self._protected_support_related(pivot, item)
                        for item in members
                    ):
                        continue
                    compacted = self._clone_groups(groups)
                    compacted[group_index] = _WorkingGroup(
                        pivot,
                        [
                            item
                            for item in members
                            if item.identifier != pivot.identifier
                        ],
                    )
                    repack_trials.append((pivot, compacted, tuple(members)))
            all_members = [*protected_selected, candidate]
            if len(groups) > 1:
                for pivot in all_members:
                    if not all(
                        item.identifier == pivot.identifier
                        or self._protected_support_related(pivot, item)
                        for item in all_members
                    ):
                        continue
                    repack_trials.append(
                        (
                            pivot,
                            [
                                _WorkingGroup(
                                    pivot,
                                    [
                                        item
                                        for item in all_members
                                        if item.identifier != pivot.identifier
                                    ],
                                )
                            ],
                            tuple(all_members),
                        )
                    )
            seen_repack_layouts: set[tuple[Any, ...]] = set()
            for pivot, compacted, repacked_members in repack_trials:
                layout = tuple(
                    (
                        group.anchor.identifier,
                        tuple(item.identifier for item in group.supports),
                    )
                    for group in compacted
                )
                if layout in seen_repack_layouts:
                    continue
                seen_repack_layouts.add(layout)
                if not self._fits(
                    compacted,
                    query_time_seconds=incident.query_time_seconds,
                    request=request,
                ):
                    continue
                placement_options.append(
                    (
                        self._bundle_cost(
                            compacted,
                            query_time_seconds=incident.query_time_seconds,
                            request=request,
                        ),
                        2,
                        "compacted",
                        compacted,
                        {
                            "reason": "shared_group_envelope_compaction",
                            "anchor": pivot.identifier,
                            "observation_ids": [
                                pivot.identifier,
                                *(
                                    item.identifier
                                    for item in repacked_members
                                    if item.identifier != pivot.identifier
                                ),
                            ],
                        },
                    )
                )
            fitting_trial: list[_WorkingGroup] | None = None
            placement_kind: str | None = None
            repack_diagnostic: dict[str, Any] | None = None
            if placement_options:
                _, _, placement_kind, fitting_trial, repack_diagnostic = min(
                    placement_options,
                    key=lambda item: (item[0], item[1]),
                )
            candidate_is_anchor = bool(
                fitting_trial
                and any(
                    group.anchor.identifier == candidate.identifier
                    for group in fitting_trial
                )
            )
            if not candidate_is_anchor:
                protected_anchor_deferrals.append(
                    {
                        "observation_id": candidate.identifier,
                        "reason": (
                            "protected_anchor_higher_token_cost"
                            if anchor_fits
                            else "protected_anchor_envelope_overflow"
                        ),
                        "priority": {
                            "severity": candidate.signals.severity,
                            "target_specific": candidate.signals.target_bearing,
                            "recency": candidate.observation.metadata.event_end_time_seconds,
                            "diagnostic_score": candidate.final_score,
                        },
                    }
                )
            if placement_kind == "compacted" and repack_diagnostic is not None:
                protected_repacks.append(repack_diagnostic)
            if fitting_trial is not None:
                groups = fitting_trial
                selected_ids.add(candidate.identifier)
                protected_selected.append(candidate)
                candidate.selected_as = (
                    "anchor"
                    if any(
                        group.anchor.identifier == candidate.identifier
                        for group in groups
                    )
                    else "support"
                )
                candidate.selection_phase = "protected"
                candidate.remaining_budget_at_decision = (
                    self.config.total_token_budget
                    - self._bundle_cost(
                        groups,
                        query_time_seconds=incident.query_time_seconds,
                        request=request,
                    )
                )
                continue
            candidate.rejection_reason = "protected_budget_overflow"
            candidate.hard_excluded = True
            candidate.remaining_budget_at_decision = (
                self.config.total_token_budget
                - self._bundle_cost(
                    groups,
                    query_time_seconds=incident.query_time_seconds,
                    request=request,
                )
            )
            protected_omissions.append(
                {
                    "observation_id": candidate.identifier,
                    "protection_reasons": list(candidate.protected_reasons),
                    "protection_reason": candidate.protected_reasons[0],
                    "equivalence_group_id": (
                        candidate.signals.equivalence_group_id
                    ),
                    "reason": "protected_budget_overflow",
                    "rank": protected_rank,
                    "estimated_token_cost": candidate.estimated_cost,
                    "selected_representative": (
                        candidate.equivalence_representative_id
                    ),
                    "remaining_budget": candidate.remaining_budget_at_decision,
                    "priority": {
                        "linkage_strength": (
                            candidate.signals.alert_linkage_strength
                        ),
                        "estimated_target_role": (
                            candidate.signals.estimated_target_role
                        ),
                        "severity": candidate.signals.severity,
                        "target_specific": candidate.signals.target_bearing,
                        "recency": candidate.observation.metadata.event_end_time_seconds,
                        "diagnostic_score": candidate.final_score,
                    },
                }
            )

        # Learned quality-diversity remains the ordinary anchor selector.  The
        # protected layer above is a bounded safety admission outside A*.
        normal_pool = [
            item
            for item in reranked_top_m
            if item.identifier not in selected_ids
            and not item.hard_excluded
            and (not constrain_to_matches or item.signals.request_match)
        ]
        normal_anchor_tokens = 0
        normal_anchor_count = 0
        selected_normal: list[_Candidate] = []
        while normal_pool and normal_anchor_count < self.config.max_anchors:
            choices: list[tuple[float, str, _Candidate]] = []
            for candidate in normal_pool:
                if (
                    normal_anchor_tokens + candidate.estimated_cost
                    > self.config.anchor_token_budget
                ):
                    continue
                penalty = sum(
                    max(
                        0.0,
                        float(torch.dot(candidate.aspect, prior.aspect).item())
                        - self.config.diversity_threshold,
                    )
                    for prior in selected_normal
                )
                gain = candidate.final_score - self.config.diversity_weight * penalty
                choices.append((gain, candidate.identifier, candidate))
            if not choices:
                break
            gain, _, candidate = min(choices, key=lambda item: (-item[0], item[1]))
            normal_pool.remove(candidate)
            if gain < self.config.minimum_anchor_gain:
                candidate.rejection_reason = "below_minimum_anchor_gain"
                break
            trial = self._trial_with_anchor(groups, candidate)
            if not self._fits(
                trial,
                query_time_seconds=incident.query_time_seconds,
                request=request,
            ):
                candidate.rejection_reason = "no_candidate_fits"
                continue
            groups = trial
            selected_ids.add(candidate.identifier)
            selected_normal.append(candidate)
            normal_anchor_tokens += candidate.estimated_cost
            normal_anchor_count += 1
            candidate.selected_as = "anchor"
            candidate.selection_phase = "quality_diversity"

        # Build one deterministic ANN over the complete eligible corpus and
        # globally order support edges so early anchors cannot monopolize K/B.
        if groups and len(candidates) > 1:
            corpus = sorted(candidates, key=lambda item: item.identifier)
            corpus_vectors = torch.stack([item.aspect for item in corpus])
            corpus_ids = tuple(item.identifier for item in corpus)
            if self.neighbor_index_factory is None:
                neighbor_index: NeighborIndex = ProjectedANNIndex(
                    corpus_vectors,
                    corpus_ids,
                    projection_count=self.config.ann_projection_count,
                    candidate_multiplier=self.config.ann_candidate_multiplier,
                    seed=self.config.ann_seed,
                )
            else:
                neighbor_index = self.neighbor_index_factory(corpus_vectors, corpus_ids)
            position_by_id = {
                item.identifier: position for position, item in enumerate(corpus)
            }
            requested_count = max(
                self.config.supports_per_anchor,
                self.config.supports_per_anchor * self.config.ann_candidate_multiplier,
            )
            edges: list[tuple[float, str, int, _Candidate]] = []
            for group_index, group in enumerate(groups):
                anchor = group.anchor
                approximate_positions = set(
                    neighbor_index.query(
                        position_by_id[anchor.identifier],
                        requested_count,
                        exclude=frozenset(
                            position_by_id[item_id]
                            for item_id in selected_ids
                            if item_id in position_by_id
                        ),
                    )
                )
                for position, candidate in enumerate(corpus):
                    if (
                        candidate.identifier not in selected_ids
                        and candidate.identifier != anchor.identifier
                    ):
                        explicitly_related = (
                            bool(
                                set(anchor.signals.entity_ids)
                                & set(candidate.signals.entity_ids)
                            )
                            or (
                                anchor.observation.metadata.primary_subsystem
                                == candidate.observation.metadata.primary_subsystem
                            )
                            or candidate.signals.request_match
                        )
                    else:
                        explicitly_related = False
                    if position not in approximate_positions and not explicitly_related:
                        continue
                    if candidate.identifier in selected_ids:
                        continue
                    if candidate.hard_excluded:
                        continue
                    if constrain_to_matches and not (
                        candidate.signals.request_match or candidate.protected
                    ):
                        continue
                    value = self._support_value(anchor, candidate)
                    if value < self.config.minimum_support_score:
                        candidate.rejection_reason = "below_minimum_support_score"
                        continue
                    edges.append((value, candidate.identifier, group_index, candidate))
            edges.sort(key=lambda item: (-item[0], item[1], item[2]))
            for _, _, group_index, candidate in edges:
                if candidate.identifier in selected_ids:
                    continue
                if len(groups[group_index].supports) >= self.config.supports_per_anchor:
                    continue
                trial = self._trial_with_support(groups, group_index, candidate)
                if not self._fits(
                    trial,
                    query_time_seconds=incident.query_time_seconds,
                    request=request,
                ):
                    continue
                groups = trial
                selected_ids.add(candidate.identifier)
                candidate.selected_as = "support"
                candidate.selection_phase = "corroboration"

        # Non-oracle coverage guard over observable categories only.
        coverage_rules: tuple[tuple[str, Any], ...] = (
            ("active_alert", lambda item: item.signals.active_alert),
            ("violated_slo_or_health", lambda item: item.signals.slo_violation),
            (
                "target_bearing",
                lambda item: item.signals.target_bearing
                and (item.signals.active_alert or item.signals.anomalous_metric),
            ),
            ("anomalous_metric", lambda item: item.signals.anomalous_metric),
            ("targeted_request", lambda item: item.signals.request_match),
        )
        coverage_added: list[str] = []
        coverage_evictions: list[dict[str, Any]] = []

        def coverage_placement_trials(
            base_groups: Sequence[_WorkingGroup], candidate: _Candidate
        ) -> list[_WorkingGroup]:
            """Try shared-envelope support placement before a new anchor."""

            trials = self._ordered_support_trials(base_groups, candidate)
            trials.append(self._trial_with_anchor(base_groups, candidate))
            return trials

        for coverage_index, (coverage_name, predicate) in enumerate(coverage_rules):
            available = [
                item
                for item in candidates
                if predicate(item)
                and not item.hard_excluded
            ]
            if not available or any(
                item.identifier in selected_ids for item in available
            ):
                continue
            ordered_fallbacks = sorted(
                available,
                key=self._protected_rank,
            )
            coverage_satisfied = False
            for candidate in ordered_fallbacks:
                if candidate.identifier in selected_ids:
                    break
                for trial in coverage_placement_trials(groups, candidate):
                    if not self._fits(
                        trial,
                        query_time_seconds=incident.query_time_seconds,
                        request=request,
                    ):
                        continue
                    groups = trial
                    selected_ids.add(candidate.identifier)
                    candidate.selected_as = "fallback"
                    candidate.selection_phase = "coverage_guard"
                    coverage_added.append(coverage_name)
                    coverage_satisfied = True
                    break
                if coverage_satisfied:
                    break
            if coverage_satisfied:
                continue

            # If ordinary QD/support rows consumed the budget, the coverage
            # guard may deterministically evict the least useful unprotected
            # rows.  Protected evidence and earlier coverage fallbacks are
            # never displaced.
            for candidate in ordered_fallbacks:
                if candidate.identifier in selected_ids:
                    coverage_satisfied = True
                    break
                victims = sorted(
                    (
                        item
                        for group in groups
                        for item in (group.anchor, *group.supports)
                        if not item.protected
                        and item.selection_phase != "coverage_guard"
                    ),
                    key=lambda item: (
                        -int(
                            item.redundant
                            or item.signals.zero_series
                            or item.signals.static_config
                            or item.signals.bookkeeping_log
                        ),
                        item.final_score,
                        item.observation.metadata.event_end_time_seconds,
                        item.identifier,
                    ),
                )
                trial_groups = self._clone_groups(groups)
                evicted: list[_Candidate] = []
                for victim in victims:
                    before_ids = {
                        item.identifier
                        for current_group in trial_groups
                        for item in (current_group.anchor, *current_group.supports)
                    }
                    prospective_groups = self._without_candidate(
                        trial_groups, victim.identifier
                    )
                    placement_trials = coverage_placement_trials(
                        prospective_groups, candidate
                    )
                    trial_ids = {
                        item.identifier
                        for trial_group in placement_trials[0]
                        for item in (trial_group.anchor, *trial_group.supports)
                    }
                    if any(
                        any(
                            prior_predicate(item)
                            for item in candidates
                            if item.identifier in before_ids
                        )
                        and not any(
                            prior_predicate(item)
                            for item in candidates
                            if item.identifier in trial_ids
                        )
                        for _, prior_predicate in coverage_rules[:coverage_index]
                    ):
                        # This victim is the sole representative of an earlier
                        # observable coverage class. Skip it rather than
                        # destructively carrying the failed removal into the
                        # next trial.
                        continue
                    trial_groups = prospective_groups
                    evicted.append(victim)
                    fitting_trial = next(
                        (
                            trial
                            for trial in placement_trials
                            if self._fits(
                                trial,
                                query_time_seconds=incident.query_time_seconds,
                                request=request,
                            )
                        ),
                        None,
                    )
                    if fitting_trial is None:
                        continue
                    groups = fitting_trial
                    for removed in evicted:
                        selected_ids.discard(removed.identifier)
                        removed.selected_as = None
                        removed.selection_phase = None
                        removed.eviction_reason = (
                            f"evicted_for_coverage:{coverage_name}"
                        )
                        removed.rejection_reason = removed.eviction_reason
                        coverage_evictions.append(
                            {
                                "observation_id": removed.identifier,
                                "reason": removed.eviction_reason,
                            }
                        )
                    selected_ids.add(candidate.identifier)
                    candidate.selected_as = "fallback"
                    candidate.selection_phase = "coverage_guard"
                    candidate.rejection_reason = None
                    coverage_added.append(coverage_name)
                    coverage_satisfied = True
                    break
                if coverage_satisfied:
                    break

        # Count caps no longer terminate packing.  Scan all useful remaining
        # observations in deterministic marginal-utility order and attach each
        # to the most compatible group that fits the exact shared serialization.
        fill_eligible = [
            item
            for item in candidates
            if item.identifier not in selected_ids
            and not item.hard_excluded
            and (
                not targeted
                or not request_matches
                or item.signals.request_match
                or item.protected
            )
        ]
        useful_nonfitting: list[_Candidate] = []
        below_fill_utility: list[_Candidate] = []
        remaining_fill = list(fill_eligible)
        while remaining_fill:
            selected_channels = {
                item.observation.channel.value
                for group in groups
                for item in (group.anchor, *group.supports)
            }
            selected_subsystems = {
                item.observation.metadata.primary_subsystem
                for group in groups
                for item in (group.anchor, *group.supports)
            }
            ranked_remaining: list[tuple[float, str, _Candidate]] = []
            for candidate in remaining_fill:
                marginal_utility = candidate.final_score
                if candidate.observation.channel.value not in selected_channels:
                    marginal_utility += self.config.cross_modal_bonus
                if (
                    candidate.observation.metadata.primary_subsystem
                    not in selected_subsystems
                ):
                    marginal_utility += self.config.support_subsystem_weight
                ranked_remaining.append(
                    (marginal_utility, candidate.identifier, candidate)
                )
            ranked_remaining.sort(key=lambda item: (-item[0], item[1]))
            admitted_this_round = False
            for rank_index, (marginal_utility, _, candidate) in enumerate(
                ranked_remaining
            ):
                remaining_fill = [
                    item
                    for item in remaining_fill
                    if item.identifier != candidate.identifier
                ]
                if marginal_utility < self.config.minimum_budget_fill_score:
                    candidate.rejection_reason = "below_budget_fill_utility"
                    below_fill_utility.append(candidate)
                    # Every remaining row has no greater utility for the
                    # current bundle.  Mark all explicitly and terminate.
                    for _, _, lower_candidate in ranked_remaining[rank_index + 1 :]:
                        remaining_fill = [
                            item
                            for item in remaining_fill
                            if item.identifier != lower_candidate.identifier
                        ]
                        lower_candidate.rejection_reason = "below_budget_fill_utility"
                        below_fill_utility.append(lower_candidate)
                    break
                placements = sorted(
                    range(len(groups)),
                    key=lambda group_index: (
                        -self._support_value(groups[group_index].anchor, candidate),
                        groups[group_index].anchor.identifier,
                    ),
                )
                admitted = False
                if not placements:
                    trial = self._trial_with_anchor(groups, candidate)
                    if self._fits(
                        trial,
                        query_time_seconds=incident.query_time_seconds,
                        request=request,
                    ):
                        groups = trial
                        candidate.selected_as = "anchor"
                        admitted = True
                else:
                    for group_index in placements:
                        trial = self._trial_with_support(groups, group_index, candidate)
                        if not self._fits(
                            trial,
                            query_time_seconds=incident.query_time_seconds,
                            request=request,
                        ):
                            continue
                        groups = trial
                        candidate.selected_as = "support"
                        admitted = True
                        break
                if admitted:
                    selected_ids.add(candidate.identifier)
                    candidate.selection_phase = "budget_fill"
                    candidate.rejection_reason = None
                    if candidate.eviction_reason is not None:
                        coverage_evictions = [
                            item
                            for item in coverage_evictions
                            if item["observation_id"] != candidate.identifier
                        ]
                        candidate.eviction_reason = None
                    admitted_this_round = True
                    break
                useful_nonfitting.append(candidate)
                # This row was considered by the final packing phase and did
                # not fit.  That is the operative rejection reason even if an
                # earlier support cutoff also rejected it.
                # ``eviction_reason`` preserves the historical coverage
                # displacement; this field records the final retry outcome.
                candidate.rejection_reason = "no_candidate_fits"
            if not admitted_this_round:
                break

        # Coverage eviction can remove an anchor and promote its first support.
        # Reconcile audit/retention roles with the final group structure before
        # assigning rejection reasons or committing episode memory.
        final_anchor_ids = {group.anchor.identifier for group in groups}
        for candidate in candidates:
            if candidate.identifier not in selected_ids:
                continue
            if candidate.identifier in final_anchor_ids:
                candidate.selected_as = "anchor"
            else:
                candidate.selected_as = "support"

        # Assign explicit final rejection reasons to every scored observation.
        for candidate in candidates:
            if candidate.identifier in selected_ids:
                candidate.rejection_reason = None
            elif candidate.rejection_reason is None:
                if targeted and request_matches and not candidate.signals.request_match:
                    candidate.rejection_reason = "outside_targeted_request"
                elif candidate.redundant:
                    candidate.rejection_reason = "excessive_semantic_redundancy"
                elif candidate.identifier not in learned_top_m:
                    candidate.rejection_reason = "outside_top_m_not_admitted"
                else:
                    candidate.rejection_reason = "not_selected"

        used_tokens = self._bundle_cost(
            groups,
            query_time_seconds=incident.query_time_seconds,
            request=request,
        )
        target_scope_ambiguity, target_scope_candidates = (
            self._target_scope_summary(groups)
        )
        if used_tokens > self.config.total_token_budget:
            raise RuntimeError("StateBundle packed evidence exceeds total budget")
        unused_tokens = self.config.total_token_budget - used_tokens
        for candidate in (*all_current, *all_retained):
            if candidate.remaining_budget_at_decision is None:
                candidate.remaining_budget_at_decision = unused_tokens
        for omission in protected_omissions:
            omission.setdefault("remaining_budget", unused_tokens)
        # The benchmark renderer is fixed to o200k_base. Always compute that
        # serialization independently, even when a caller injects another
        # tokenizer or synthetic cost estimator for an experiment.
        exact_estimator = CompactAgentTokenEstimator(encoding_name="o200k_base")
        exact_serialized_tokens = exact_estimator.estimate_bundle(
            tuple(
                (
                    group.anchor.observation,
                    tuple(item.observation for item in group.supports),
                )
                for group in groups
            ),
            self.config.redaction_policy,
            query_time_seconds=incident.query_time_seconds,
            request=request,
            target_scope_ambiguity=target_scope_ambiguity,
            target_scope_candidates=target_scope_candidates,
        )
        exact_accounting = (
            isinstance(self.token_estimator, CompactAgentTokenEstimator)
            and self.token_estimator.encoding_name == "o200k_base"
        )
        if exact_accounting and exact_serialized_tokens != used_tokens:
            raise RuntimeError("exact StateBundle token accounting diverged")
        if unused_tokens == 0:
            unused_reason = "budget_reached"
        elif useful_nonfitting or protected_omissions:
            unused_reason = "no_candidate_fits"
        elif below_fill_utility:
            unused_reason = "no_useful_candidate"
        else:
            unused_reason = "all_candidates_exhausted"

        output_groups: list[CorroboratingEvidenceGroup] = []
        for group in groups:
            anchor = SelectedAnchor(
                observation=group.anchor.observation,
                observation_index=group.anchor.observation_index,
                token_cost=group.anchor.estimated_cost,
                relevance_score=group.anchor.learned_score,
            )
            output_groups.append(
                CorroboratingEvidenceGroup(
                    anchor=anchor,
                    evidence=tuple(item.observation for item in group.supports),
                    token_cost=(
                        group.anchor.estimated_cost
                        + sum(item.estimated_cost for item in group.supports)
                    ),
                )
            )

        selected_candidates = [
            item for item in candidates if item.identifier in selected_ids
        ]
        protected_omissions = [
            item
            for item in protected_omissions
            if item["observation_id"] not in selected_ids
        ]
        selected_request_matches = sum(
            item.signals.request_match for item in selected_candidates
        )
        request_status: dict[str, Any] = {}
        if targeted:
            if not request_matches:
                status = "no_matching_canonical_observation"
                message = (
                    "No canonical observation matched the requested telemetry filters; "
                    "narrow or verify metric, entity, subsystem, alert, and lookback."
                )
            elif selected_request_matches < len(request_matches):
                status = "narrow_request"
                message = (
                    "The targeted raw result exceeded useful budget capacity; narrow "
                    "metric, entity, subsystem, alert, or lookback scope."
                )
            else:
                status = "satisfied"
                message = (
                    "All unique matching targeted facts that remained eligible fit; "
                    "exact duplicate rows were suppressed."
                )
            request_status = {
                "status": status,
                "message": message,
                "matched_observation_count": canonical_request_match_count,
                "unique_matched_fact_count": len(request_matches),
                "duplicate_matched_observation_count": (
                    canonical_request_match_count - len(request_matches)
                ),
                "selected_match_count": selected_request_matches,
                "requested_metric_names": list(request.metric_names if request else ()),
                "requested_entity_ids": list(request.entity_ids if request else ()),
                "requested_subsystem_ids": list(
                    getattr(request, "subsystem_ids", ()) if request else ()
                ),
                "requested_alert_names": list(
                    getattr(request, "alert_names", ()) if request else ()
                ),
                "lookback_seconds": request.lookback_seconds if request else 300.0,
            }

        previous_selected = set(memory.last_selected_logical_keys)
        current_selected = {item.signals.logical_key for item in selected_candidates}
        retained_between_cuts = sorted(previous_selected & current_selected)
        newly_admitted = sorted(current_selected - previous_selected)
        selection_evicted = (
            sorted(previous_selected - current_selected) if causal_cut_advanced else []
        )
        known_eviction_reasons = {
            item["logical_key"]: item["reason"] for item in memory_evictions
        }
        consecutive_evictions = [
            {
                "logical_key": key,
                "reason": known_eviction_reasons.get(key, "selection_not_retained"),
            }
            for key in selection_evicted
        ]

        # Commit only compact important evidence after successful selection.
        if memory.active:
            stably_released_keys = set(memory.stable_resolution_tombstones)
            for candidate in selected_candidates:
                if (
                    candidate.signals.logical_key in stably_released_keys
                    and not candidate.signals.active_alert
                    and not candidate.signals.slo_violation
                ):
                    # A resolved row may still rank highly in the release cut;
                    # do not immediately turn it back into ordinary sticky
                    # evidence after the stability criterion was satisfied.
                    continue
                if (
                    candidate.observation.channel.value == "alert"
                    and not candidate.signals.active_alert
                    and not candidate.signals.slo_violation
                    and not candidate.post_action
                ):
                    continue
                important = bool(
                    candidate.selected_as == "anchor"
                    or candidate.selection_phase == "coverage_guard"
                    or candidate.protected
                    or candidate.signals.active_alert
                    or candidate.signals.slo_violation
                    or candidate.signals.target_bearing
                    or (
                        any(
                            self._is_action_evidence(candidate, evaluation_action)
                            for evaluation_action in evaluation_actions
                        )
                        if post_action_active
                        else self._is_action_evidence(candidate)
                    )
                )
                if not important or candidate.retained and candidate.post_action:
                    continue
                unresolved_kind = (
                    "alert"
                    if candidate.signals.active_alert
                    else ("slo" if candidate.signals.slo_violation else None)
                )
                existing = memory.retained.get(candidate.signals.logical_key)
                if (
                    unresolved_kind is None
                    and existing is not None
                    and existing.unresolved_kind is not None
                    and memory.resolution_streaks.get(candidate.signals.logical_key, 0)
                    < self.config.resolution_stability_cuts
                ):
                    # Keep the unresolved identity through the configured
                    # stable-resolution window.  Otherwise the first resolved
                    # row would accidentally convert it to ordinary sticky
                    # evidence and postpone release.
                    unresolved_kind = existing.unresolved_kind
                expires_cut = (
                    existing.expires_cut
                    if existing is not None and existing.unresolved_kind is None
                    else cut_index + self.config.retention_max_cuts
                )
                if unresolved_kind:
                    expires_cut = cut_index + self.config.retention_max_cuts
                elif candidate.post_action:
                    expires_cut = cut_index + self.config.post_action_retention_cuts
                memory.retained[candidate.signals.logical_key] = _RetainedEvidence(
                    memory_key=candidate.signals.logical_key,
                    observation=candidate.observation,
                    aspect=candidate.aspect.detach().cpu().clone(),
                    learned_score=candidate.learned_score,
                    signals=candidate.signals,
                    estimated_cost=candidate.estimated_cost,
                    components=candidate.components,
                    final_score=candidate.final_score,
                    admitted_cut=(
                        existing.admitted_cut if existing is not None else cut_index
                    ),
                    expires_cut=expires_cut,
                    unresolved_kind=unresolved_kind,
                )
            if len(memory.retained) > self.config.max_retained_observations:
                retained_priority = sorted(
                    memory.retained.items(),
                    key=lambda item: (
                        -int(item[1].unresolved_kind is not None),
                        -int(item[1].pre_action),
                        -item[1].admitted_cut,
                        -item[1].final_score,
                        item[0],
                    ),
                )
                for memory_key, entry in retained_priority[
                    self.config.max_retained_observations :
                ]:
                    memory.retained.pop(memory_key, None)
                    memory.resolution_streaks.pop(memory_key, None)
                    memory_evictions.append(
                        {
                            "logical_key": entry.signals.logical_key,
                            "observation_id": entry.observation.observation_id,
                            "reason": "memory_capacity",
                        }
                    )
            if (
                len(memory.stable_resolution_tombstones)
                > self.config.max_retained_observations
            ):
                oldest_tombstones = sorted(
                    memory.stable_resolution_tombstones.items(),
                    key=lambda item: (item[1], item[0]),
                )
                for logical_key, _ in oldest_tombstones[
                    : -self.config.max_retained_observations
                ]:
                    memory.stable_resolution_tombstones.pop(logical_key, None)
            memory.cut_index = cut_index
            memory.causal_cut_key = causal_cut_key
            if causal_cut_advanced:
                memory.last_selected_logical_keys = current_selected
            else:
                memory.last_selected_logical_keys |= current_selected

        active_alert_keys = {
            signal.logical_key for signal in signals if signal.active_alert
        }
        active_slo_keys = {
            signal.logical_key for signal in signals if signal.slo_violation
        }
        selected_current_logical_keys = {
            item.signals.logical_key
            for item in selected_candidates
            if not item.retained
        }
        candidates_for_audit = sorted(
            [
                *all_current,
                *all_retained,
            ],
            key=lambda item: (item.observation_index, item.identifier),
        )
        diagnostics = {
            "policy_version": "observable-safeguards-v2",
            "inference_policy_parameters": {
                item.name: getattr(self.config, item.name)
                for item in dataclass_fields(self.config)
                if isinstance(
                    getattr(self.config, item.name),
                    (bool, int, float, str, type(None)),
                )
            },
            "configured_budget": self.config.total_token_budget,
            "budget_accounted_telemetry_tokens": used_tokens,
            "actual_serialized_telemetry_tokens": exact_serialized_tokens,
            "token_accounting_method": type(self.token_estimator).__name__,
            "token_accounting_encoding": getattr(
                self.token_estimator, "encoding_name", None
            ),
            "actual_serialization_encoding": "o200k_base",
            "token_accounting_exact": exact_accounting,
            "token_budget_semantics": (
                "serialized_tokens" if exact_accounting else "custom_estimator_units"
            ),
            "estimated_singleton_tokens": sum(
                item.estimated_cost for item in selected_candidates
            ),
            "singleton_cost_semantics": (
                "per_candidate_estimate_not_additive_joint_bundle_cost"
            ),
            "group_singleton_estimated_tokens": [
                {
                    "group": index,
                    "anchor": group.anchor.observation.observation_id,
                    "estimated_tokens": group.token_cost,
                }
                for index, group in enumerate(output_groups)
            ],
            "utilization_ratio": used_tokens / self.config.total_token_budget,
            "unused_tokens": unused_tokens,
            "actual_serialized_utilization_ratio": (
                exact_serialized_tokens / self.config.total_token_budget
            ),
            "actual_serialized_unused_tokens": (
                self.config.total_token_budget - exact_serialized_tokens
            ),
            "actual_serialized_within_budget": (
                exact_serialized_tokens <= self.config.total_token_budget
            ),
            "unused_capacity_reason": unused_reason,
            "input_observation_count": count,
            "unique_candidate_count": len(candidates),
            "learned_top_m_count": len(learned_top_m),
            "protected_observation_count": sum(item.protected for item in candidates),
            "protected_anchor_deferrals": protected_anchor_deferrals,
            "protected_group_repacks": protected_repacks,
            "protected_omissions": protected_omissions,
            "coverage_fallbacks": coverage_added,
            "coverage_evictions": coverage_evictions,
            "targeted_request": targeted,
            "target_scope_ambiguity": target_scope_ambiguity,
            "target_scope_candidates": list(target_scope_candidates),
            "learned_top_m_coverage_misses": [
                {
                    "observation_id": item.identifier,
                    "selection_phase": item.selection_phase,
                    "protection_reasons": list(item.protected_reasons),
                }
                for item in selected_candidates
                if item.identifier not in learned_top_m
                and item.selection_phase in {"protected", "coverage_guard"}
            ],
            "selected_outside_learned_top_m": [
                {
                    "observation_id": item.identifier,
                    "selection_phase": item.selection_phase,
                    "final_ranking_score": item.final_score,
                    "observable_features": {
                        "active_alert": item.signals.active_alert,
                        "target_bearing": item.signals.target_bearing,
                        "slo_violation": item.signals.slo_violation,
                        "anomalous_metric": item.signals.anomalous_metric,
                        "recent_config_change": item.signals.recent_config_change,
                        "request_match": item.signals.request_match,
                    },
                }
                for item in selected_candidates
                if item.identifier not in learned_top_m
            ],
            "candidates": [
                self._candidate_diagnostic(item) for item in candidates_for_audit
            ],
            "causal_cut_transition": {
                "cut_index": cut_index,
                "causal_cut_advanced": causal_cut_advanced,
                "retained_observations": retained_between_cuts,
                "newly_admitted_observations": newly_admitted,
                "evicted_observations": consecutive_evictions,
                "memory_evictions": memory_evictions,
                "stable_resolution_tombstones": sorted(
                    memory.stable_resolution_tombstones
                ),
                "active_unresolved_alert_count": len(active_alert_keys),
                "unresolved_alerts_retained": (
                    active_alert_keys <= selected_current_logical_keys
                    if active_alert_keys
                    else None
                ),
                "active_unresolved_slo_count": len(active_slo_keys),
                "unresolved_slo_evidence_retained": (
                    active_slo_keys <= selected_current_logical_keys
                    if active_slo_keys
                    else None
                ),
                "recorded_actions": list(memory.actions),
                "active_evaluation_actions": [
                    dict(item) for item in evaluation_actions
                ],
                "active_action_sequences": [
                    int(item["sequence"]) for item in evaluation_actions
                ],
            },
        }
        result = StateBundleOutput(
            incident_id=incident.incident_id,
            query_time_seconds=incident.query_time_seconds,
            snapshot_id=incident.snapshot_id,
            groups=tuple(output_groups),
            token_budget=self.config.total_token_budget,
            used_tokens=used_tokens,
            candidate_count=max(len(reranked_top_m), len(output_groups)),
            redaction_policy=self.config.redaction_policy,
            request_status=request_status,
            target_scope_ambiguity=target_scope_ambiguity,
            target_scope_candidates=target_scope_candidates,
            selection_audit=diagnostics,
        )
        _assert_agent_safe(result.to_dict())
        if memory.active:
            memory.cache_key = cache_key
            memory.cached_output = result
            self._memory = memory
        return result


__all__ = [
    "CharacterTokenEstimator",
    "CompactAgentTokenEstimator",
    "NeighborIndex",
    "ProjectedANNIndex",
    "StateBundleInference",
    "TokenEstimator",
]
