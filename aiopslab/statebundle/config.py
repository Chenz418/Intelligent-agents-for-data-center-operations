r"""Validated, serializable configuration for StateBundle.

The paper leaves backbone sizes, ANN details, token
estimation, and the exact corroboration score :math:`\kappa` unspecified.  They
are explicit here so those assumptions remain replaceable rather than being
hidden in model or inference code.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
import math
from pathlib import Path
from typing import Any

import yaml

from aiopslab.statebundle.types import RedactionPolicy, RootCauseCatalog


def _positive(value: float | int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    if not math.isfinite(float(value)) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


def _non_negative(value: float | int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    if not math.isfinite(float(value)) or value < 0:
        raise ValueError(f"{name} must be finite and non-negative")


def _probability(value: float, name: str) -> None:
    if not 0.0 <= value < 1.0:
        raise ValueError(f"{name} must be in [0, 1)")


def _strict_dataclass(cls: type[Any], raw: Mapping[str, Any], name: str) -> Any:
    allowed = set(cls.__dataclass_fields__)
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"unknown {name} fields: {unknown}")
    return cls(**raw)


def _plain(value: Any) -> Any:
    """Convert frozen dataclasses/mapping proxies/enums to YAML-safe values."""

    if is_dataclass(value) and not isinstance(value, type):
        return {item.name: _plain(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, Mapping):
        return {
            (key.value if isinstance(key, Enum) else str(key)): _plain(item)
            for key, item in value.items()
        }
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, Enum):
        return value.value
    return value


@dataclass(frozen=True, slots=True)
class StateBundleModelConfig:
    """Architecture and differentiable selector hyperparameters."""

    encoder_dimension: int = 128
    shared_dimension: int = 128
    relevance_dimension: int = 64
    aspect_dimension: int = 64
    attention_dimension: int = 64
    propagation_hidden_dimension: int = 128
    hash_vocabulary_size: int = 16_384
    max_text_tokens: int = 128
    metric_sequence_length: int = 128
    metric_normalization_epsilon: float = 1e-6
    metric_clip_value: float = 20.0
    # Checkpoint identity fields are validated against the bundled weights.
    observation_dropout: float = 0.1
    propagation_dropout: float = 0.1
    global_relevance_weight: float = 1.0
    subsystem_relevance_weight: float = 1.0
    propagation_relevance_weight: float = 0.5
    relevance_threshold: float = 0.0
    gate_temperature: float = 0.2
    relevance_temperature: float = 0.2
    root_cause_temperature: float = 0.1
    normalization_epsilon: float = 1e-8

    def __post_init__(self) -> None:
        for name in (
            "encoder_dimension",
            "shared_dimension",
            "relevance_dimension",
            "aspect_dimension",
            "attention_dimension",
            "propagation_hidden_dimension",
            "hash_vocabulary_size",
            "max_text_tokens",
            "metric_sequence_length",
        ):
            _positive(getattr(self, name), name)
        if self.hash_vocabulary_size < 2:
            raise ValueError("hash_vocabulary_size must be >= 2")
        for name in (
            "metric_normalization_epsilon",
            "metric_clip_value",
            "gate_temperature",
            "relevance_temperature",
            "root_cause_temperature",
            "normalization_epsilon",
        ):
            _positive(getattr(self, name), name)
        _probability(self.observation_dropout, "observation_dropout")
        _probability(self.propagation_dropout, "propagation_dropout")
        for name in (
            "global_relevance_weight",
            "subsystem_relevance_weight",
            "propagation_relevance_weight",
        ):
            _non_negative(getattr(self, name), name)
        if not any(
            getattr(self, name) > 0
            for name in (
                "global_relevance_weight",
                "subsystem_relevance_weight",
                "propagation_relevance_weight",
            )
        ):
            raise ValueError("at least one relevance-score weight must be positive")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "StateBundleModelConfig":
        return _strict_dataclass(cls, value, "model config")


@dataclass(frozen=True, slots=True)
class StateBundleInferenceConfig:
    """Budgeted two-stage retrieval and corroboration configuration."""

    top_m: int = 100
    anchor_token_budget: int = 1_024
    total_token_budget: int = 4_096
    max_anchors: int = 8
    supports_per_anchor: int = 4
    diversity_weight: float = 0.25
    diversity_threshold: float = 0.7
    minimum_anchor_gain: float = 0.0
    support_aspect_weight: float = 1.0
    support_relevance_weight: float = 0.5
    support_temporal_weight: float = 0.25
    support_subsystem_weight: float = 0.25
    cross_modal_bonus: float = 0.1
    temporal_scale_seconds: float = 300.0
    ann_projection_count: int = 8
    ann_candidate_multiplier: int = 8
    ann_seed: int = 2026
    characters_per_token: float = 4.0
    # Observable-only deterministic inference safeguards.  The learned
    # relevance score remains the unit-weight primary signal; every term below
    # is computed from canonical telemetry available at the causal cut.
    learned_score_weight: float = 1.0
    active_alert_bonus: float = 0.40
    critical_alert_bonus: float = 0.80
    high_alert_bonus: float = 0.60
    warning_alert_bonus: float = 0.20
    target_role_bonus: float = 0.35
    entity_identifier_bonus: float = 0.10
    slo_violation_bonus: float = 0.50
    targeted_request_bonus: float = 2.00
    sticky_evidence_bonus: float = 0.35
    post_action_evidence_bonus: float = 0.45
    anomalous_metric_bonus: float = 0.25
    recent_config_change_bonus: float = 0.35
    normal_comparison_bonus: float = 0.10
    zero_series_penalty: float = 0.60
    constant_series_penalty: float = 0.25
    redundant_observation_penalty: float = 0.50
    static_config_penalty: float = 0.40
    stale_config_penalty: float = 0.20
    bookkeeping_log_penalty: float = 0.30
    minimum_support_score: float = 0.0
    minimum_budget_fill_score: float = -0.50
    zero_fraction_threshold: float = 0.95
    zero_absolute_epsilon: float = 1e-9
    constant_variance_epsilon: float = 1e-12
    metric_anomaly_iqr_threshold: float = 3.0
    metric_change_fraction_threshold: float = 0.10
    metric_shape_round_digits: int = 6
    alert_metric_min_shared_terms: int = 2
    # Alert/observation linkage is inference-visible and deterministic.  A
    # zero gap requires overlapping aggregate windows; a positive value can be
    # configured for telemetry transports whose alert and metric cuts are
    # known to be slightly skewed.
    alert_link_max_gap_seconds: float = 0.0
    alert_counterpart_relative_tolerance: float = 0.02
    recent_config_window_seconds: float = 60.0
    stale_config_age_seconds: float = 300.0
    zero_series_representatives: int = 2
    semantic_group_representatives: int = 2
    healthy_peer_representatives: int = 2
    resolution_stability_cuts: int = 2
    retention_max_cuts: int = 6
    post_action_retention_cuts: int = 3
    max_retained_observations: int = 32
    redaction_policy: RedactionPolicy = field(default_factory=RedactionPolicy)

    def __post_init__(self) -> None:
        for name in (
            "top_m",
            "anchor_token_budget",
            "total_token_budget",
            "max_anchors",
            "supports_per_anchor",
            "ann_projection_count",
            "ann_candidate_multiplier",
            "zero_series_representatives",
            "semantic_group_representatives",
            "healthy_peer_representatives",
            "metric_shape_round_digits",
            "alert_metric_min_shared_terms",
            "resolution_stability_cuts",
            "retention_max_cuts",
            "post_action_retention_cuts",
            "max_retained_observations",
        ):
            _positive(getattr(self, name), name)
        if self.anchor_token_budget > self.total_token_budget:
            raise ValueError("anchor_token_budget must not exceed total_token_budget")
        for name in (
            "diversity_weight",
            "minimum_anchor_gain",
            "support_aspect_weight",
            "support_relevance_weight",
            "support_temporal_weight",
            "support_subsystem_weight",
            "cross_modal_bonus",
            "learned_score_weight",
            "active_alert_bonus",
            "critical_alert_bonus",
            "high_alert_bonus",
            "warning_alert_bonus",
            "target_role_bonus",
            "entity_identifier_bonus",
            "slo_violation_bonus",
            "targeted_request_bonus",
            "sticky_evidence_bonus",
            "post_action_evidence_bonus",
            "anomalous_metric_bonus",
            "recent_config_change_bonus",
            "normal_comparison_bonus",
            "zero_series_penalty",
            "constant_series_penalty",
            "redundant_observation_penalty",
            "static_config_penalty",
            "stale_config_penalty",
            "bookkeeping_log_penalty",
        ):
            _non_negative(getattr(self, name), name)
        if not -1.0 <= self.diversity_threshold <= 1.0:
            raise ValueError("diversity_threshold must be in [-1, 1]")
        _positive(self.temporal_scale_seconds, "temporal_scale_seconds")
        _positive(self.characters_per_token, "characters_per_token")
        if not math.isfinite(self.minimum_support_score):
            raise ValueError("minimum_support_score must be finite")
        if not math.isfinite(self.minimum_budget_fill_score):
            raise ValueError("minimum_budget_fill_score must be finite")
        if not 0.0 < self.zero_fraction_threshold <= 1.0:
            raise ValueError("zero_fraction_threshold must be in (0, 1]")
        for name in (
            "zero_absolute_epsilon",
            "constant_variance_epsilon",
            "metric_change_fraction_threshold",
            "alert_link_max_gap_seconds",
        ):
            _non_negative(getattr(self, name), name)
        _probability(
            self.alert_counterpart_relative_tolerance,
            "alert_counterpart_relative_tolerance",
        )
        for name in (
            "metric_anomaly_iqr_threshold",
            "recent_config_window_seconds",
            "stale_config_age_seconds",
        ):
            _positive(getattr(self, name), name)
        if isinstance(self.ann_seed, bool) or not isinstance(self.ann_seed, int):
            raise TypeError("ann_seed must be an integer")
        if not isinstance(self.redaction_policy, RedactionPolicy):
            raise TypeError("redaction_policy must be a RedactionPolicy")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "StateBundleInferenceConfig":
        raw = dict(value)
        if "redaction_policy" in raw and not isinstance(
            raw["redaction_policy"], RedactionPolicy
        ):
            raw["redaction_policy"] = RedactionPolicy.from_dict(raw["redaction_policy"])
        return _strict_dataclass(cls, raw, "inference config")


@dataclass(frozen=True, slots=True)
class StateBundleConfig:
    """Runtime model, device, inference, and prototype configuration."""

    root_cause_catalog: RootCauseCatalog
    model: StateBundleModelConfig = field(default_factory=StateBundleModelConfig)
    device: str = "cpu"
    inference: StateBundleInferenceConfig = field(
        default_factory=StateBundleInferenceConfig
    )

    def __post_init__(self) -> None:
        if not isinstance(self.root_cause_catalog, RootCauseCatalog):
            raise TypeError("root_cause_catalog must be a RootCauseCatalog")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "StateBundleConfig":
        raw = dict(value)
        if "root_cause_catalog" not in raw:
            raise ValueError("configuration requires root_cause_catalog")
        catalog_value = raw["root_cause_catalog"]
        if not isinstance(catalog_value, RootCauseCatalog):
            if isinstance(catalog_value, Mapping):
                catalog_value = catalog_value.get("classes", ())
            raw["root_cause_catalog"] = RootCauseCatalog.from_dicts(catalog_value)
        converters = {
            "model": StateBundleModelConfig.from_dict,
            "inference": StateBundleInferenceConfig.from_dict,
        }
        for name, converter in converters.items():
            if name in raw and isinstance(raw[name], Mapping):
                raw[name] = converter(raw[name])
        return _strict_dataclass(cls, raw, "StateBundle config")

    def to_dict(self) -> dict[str, Any]:
        return _plain(self)


def load_statebundle_config(path: str | Path) -> StateBundleConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("StateBundle configuration must be a YAML/JSON object")
    return StateBundleConfig.from_dict(raw)


__all__ = [
    "StateBundleConfig",
    "StateBundleInferenceConfig",
    "StateBundleModelConfig",
    "load_statebundle_config",
]
