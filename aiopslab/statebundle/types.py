"""Typed data contracts for StateBundle.

Canonical inputs contain only telemetry available to the agent at the
causal cut. Validation rejects evaluator labels and hidden annotations.

The module intentionally has no simulator imports and no runtime PyTorch
dependency.  ``ModelOutput`` uses a type-only Tensor alias so canonicalization,
serialization, and inference-input validation remain usable in lightweight
environments.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
import json
import math
import re
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, TypeAlias


if TYPE_CHECKING:
    from torch import Tensor
else:
    Tensor = Any


JSONScalar: TypeAlias = str | int | float | bool | None
JSONValue: TypeAlias = JSONScalar | tuple["JSONValue", ...] | Mapping[str, "JSONValue"]
SequenceMarker: TypeAlias = str | int


CANONICAL_SCHEMA_VERSION = "statebundle.canonical.v1"
STATEBUNDLE_OUTPUT_SCHEMA_VERSION = "statebundle.output.v1"


_ESTIMATED_TARGET_ROLES = frozenset(
    {
        "direct_target_candidate",
        "downstream_affected_scope",
        "contextual_peer",
    }
)


class TelemetryChannel(str, Enum):
    """Telemetry modalities supported by the paper's canonical adapter."""

    LOG = "log"
    METRIC = "metric"
    ALERT = "alert"
    TRACE = "trace"
    CONFIG = "config"


class EntityRole(str, Enum):
    """Observable semantic role of an entity in an observation."""

    PRODUCER = "producer"
    TARGET = "target"
    SOURCE = "source"
    DESTINATION = "destination"
    SCOPE = "scope"
    AFFECTED = "affected"


_FORBIDDEN_OBSERVABLE_KEYS = frozenset(
    {
        "annotation",
        "annotations",
        "active_fault",
        "active_faults",
        "associated_fault_effect_ids",
        "background_label",
        "causal_role",
        "effect_id",
        "evaluator",
        "evaluator_annotation",
        "evaluator_annotations",
        "evaluator_state",
        "expected",
        "expected_diagnosis",
        "expected_mitigation",
        "fault_id",
        "fault_label_id",
        "fault_mechanism",
        "fault_target",
        "fault_type",
        "ground_truth",
        "hidden_training_annotations",
        "incident_label",
        "inference_visible_target",
        "label_provenance",
        "local_effect_or_anomaly_id",
        "observable_evidence",
        "oracle",
        "pair_label",
        "prototype_id",
        "propagation_depth",
        "propagation_membership",
        "propagation_path",
        "root_cause",
        "root_cause_label",
        "score_hint",
        "score_hints",
        "split",
        "split_assignments",
        "subsystem_supervision",
        "success_criteria",
        "symptom_family",
        "training_annotations",
        "training_labels",
    }
)


def _text(value: Any, name: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    value = value.strip()
    if not value and not allow_empty:
        raise ValueError(f"{name} must not be empty")
    return value


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return value


def _probability(value: Any, name: str) -> float:
    value = _finite_number(value, name)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return value


def _enum_value(enum_type: type[Enum], value: Any, name: str) -> Any:
    if isinstance(value, enum_type):
        return value
    try:
        return enum_type(value)
    except (TypeError, ValueError) as error:
        choices = ", ".join(repr(item.value) for item in enum_type)
        raise ValueError(f"{name} must be one of {choices}") from error


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be an object")
    if not all(isinstance(key, str) for key in value):
        raise TypeError(f"{name} keys must be strings")
    return value


def _target_scope_candidates(
    value: Any,
    *,
    selected_observation_ids: frozenset[str],
) -> tuple[Mapping[str, JSONValue], ...]:
    """Validate the agent-safe, selector-estimated target-scope summary.

    These roles are deterministic interpretations of inference-visible
    telemetry. They are deliberately distinct from the training-only
    :class:`CausalRole` labels above and may reference only evidence that is
    actually present in the public bundle.
    """

    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise TypeError("target_scope_candidates must be an array")
    normalized: list[Mapping[str, JSONValue]] = []
    seen_scopes: set[str] = set()
    for index, raw_candidate in enumerate(value):
        candidate = _mapping(
            raw_candidate,
            f"target_scope_candidates[{index}]",
        )
        expected_fields = {
            "scope",
            "estimated_role",
            "supporting_observation_ids",
        }
        if set(candidate) != expected_fields:
            raise ValueError(
                "target_scope_candidates entries must contain exactly "
                "scope, estimated_role, and supporting_observation_ids"
            )
        scope = _text(
            candidate.get("scope"),
            f"target_scope_candidates[{index}].scope",
        )
        if scope in seen_scopes:
            raise ValueError("target_scope_candidates scopes must be unique")
        seen_scopes.add(scope)
        estimated_role = _text(
            candidate.get("estimated_role"),
            f"target_scope_candidates[{index}].estimated_role",
        )
        if estimated_role not in _ESTIMATED_TARGET_ROLES:
            raise ValueError(
                "target_scope_candidates estimated_role must be one of "
                f"{sorted(_ESTIMATED_TARGET_ROLES)}"
            )
        raw_ids = candidate.get("supporting_observation_ids")
        if not isinstance(raw_ids, Sequence) or isinstance(
            raw_ids, (str, bytes, bytearray)
        ):
            raise TypeError(
                "target_scope_candidates supporting_observation_ids must be an array"
            )
        observation_ids = tuple(
            _text(
                item,
                (f"target_scope_candidates[{index}].supporting_observation_ids[]"),
            )
            for item in raw_ids
        )
        if not observation_ids:
            raise ValueError(
                "target_scope_candidates supporting_observation_ids must not be empty"
            )
        if len(set(observation_ids)) != len(observation_ids):
            raise ValueError(
                "target_scope_candidates supporting_observation_ids must be unique"
            )
        unknown_ids = set(observation_ids) - selected_observation_ids
        if unknown_ids:
            raise ValueError(
                "target_scope_candidates may reference only selected observations: "
                f"{sorted(unknown_ids)}"
            )
        frozen = _freeze_json(
            {
                "scope": scope,
                "estimated_role": estimated_role,
                "supporting_observation_ids": observation_ids,
            },
            f"target_scope_candidates[{index}]",
        )
        assert isinstance(frozen, Mapping)
        normalized.append(frozen)
    return tuple(normalized)


def _freeze_json(value: Any, name: str = "value") -> JSONValue:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{name} contains a non-finite number")
        return value
    if isinstance(value, Mapping):
        frozen: dict[str, JSONValue] = {}
        for key in sorted(value):
            if not isinstance(key, str):
                raise TypeError(f"{name} contains a non-string object key")
            frozen[key] = _freeze_json(value[key], f"{name}.{key}")
        return MappingProxyType(frozen)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(_freeze_json(item, f"{name}[]") for item in value)
    raise TypeError(
        f"{name} contains unsupported value type {type(value).__name__}; "
        "canonical payloads must be JSON-compatible"
    )


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(value[key]) for key in sorted(value)}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    if isinstance(value, Enum):
        return value.value
    return value


def _reject_hidden_keys(value: Any, path: str = "observation") -> None:
    """Reject recognizable supervision fields at the observable boundary.

    The list deliberately avoids overly broad names such as ``label`` or
    ``phase`` because those can be legitimate source payload fields.  It
    targets fields whose semantics are unambiguously training-only in the
    StateBundle contract.
    """

    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = re.sub(r"[^a-z0-9]+", "_", str(key).strip().lower()).strip("_")
            if normalized in _FORBIDDEN_OBSERVABLE_KEYS:
                raise ValueError(f"training-only field is not allowed at {path}.{key}")
            _reject_hidden_keys(item, f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            _reject_hidden_keys(item, f"{path}[{index}]")


def assert_observable_only(value: Any, path: str = "observation") -> None:
    """Validate that a JSON-like value contains no evaluator-only fields.

    This public wrapper keeps the canonical input boundary and the final
    agent-output guard on one deliberately narrow denylist.  In particular,
    it avoids rejecting generic operational payload names merely because a
    similarly named field is used internally by the learned selector.
    """

    _reject_hidden_keys(value, path)


@dataclass(frozen=True, slots=True)
class RedactionPolicy:
    """Agent-visible serialization policy.

    Source references and correlation identifiers are withheld by default.
    Payload keys can be allowlisted per channel or denied globally.  The
    trusted inference backend retains the complete canonical objects for
    feature analysis, while duplicate identity follows the configured
    agent-visible projection so hidden fields cannot change packing behavior.
    """

    include_payload: bool = True
    include_entities: bool = True
    include_entity_provenance: bool = False
    include_primary_subsystem: bool = True
    include_subsystem_provenance: bool = False
    include_correlation_ids: bool = False
    include_source_references: bool = False
    include_data_quality: bool = True
    allowed_entity_roles: tuple[EntityRole, ...] = field(
        default_factory=lambda: tuple(EntityRole)
    )
    payload_allowlist: Mapping[TelemetryChannel, tuple[str, ...]] = field(
        default_factory=lambda: MappingProxyType({})
    )
    payload_denylist: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        roles = tuple(
            _enum_value(EntityRole, role, "allowed_entity_roles")
            for role in self.allowed_entity_roles
        )
        if len(set(roles)) != len(roles):
            raise ValueError("allowed_entity_roles must not contain duplicates")
        object.__setattr__(self, "allowed_entity_roles", roles)

        raw_allowlist = _mapping(self.payload_allowlist, "payload_allowlist")
        allowlist: dict[TelemetryChannel, tuple[str, ...]] = {}
        for raw_channel, raw_keys in raw_allowlist.items():
            channel = _enum_value(
                TelemetryChannel, raw_channel, "payload_allowlist channel"
            )
            if not isinstance(raw_keys, Sequence) or isinstance(raw_keys, str):
                raise TypeError("payload_allowlist values must be sequences of keys")
            keys = tuple(_text(key, "payload allowlist key") for key in raw_keys)
            if len(set(keys)) != len(keys):
                raise ValueError("payload allowlist keys must be unique per channel")
            allowlist[channel] = keys
        object.__setattr__(
            self,
            "payload_allowlist",
            MappingProxyType(
                dict(sorted(allowlist.items(), key=lambda item: item[0].value))
            ),
        )

        denylist = tuple(
            _text(key, "payload denylist key") for key in self.payload_denylist
        )
        if len(set(denylist)) != len(denylist):
            raise ValueError("payload_denylist must not contain duplicates")
        object.__setattr__(self, "payload_denylist", denylist)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RedactionPolicy":
        raw = dict(_mapping(value, "redaction policy"))
        allowed = {field_.name for field_ in cls.__dataclass_fields__.values()}
        unknown = sorted(set(raw) - allowed)
        if unknown:
            raise ValueError(f"unknown redaction policy fields: {unknown}")
        return cls(**raw)


@dataclass(frozen=True, slots=True)
class ObservationWindow:
    start_time_seconds: float
    end_time_seconds: float
    start_inclusive: bool = True
    end_inclusive: bool = True

    def __post_init__(self) -> None:
        start = _finite_number(self.start_time_seconds, "start_time_seconds")
        end = _finite_number(self.end_time_seconds, "end_time_seconds")
        if start < 0.0:
            raise ValueError("start_time_seconds must be non-negative")
        if end < start:
            raise ValueError("end_time_seconds must be >= start_time_seconds")
        if not isinstance(self.start_inclusive, bool) or not isinstance(
            self.end_inclusive, bool
        ):
            raise TypeError("window inclusivity flags must be booleans")
        object.__setattr__(self, "start_time_seconds", start)
        object.__setattr__(self, "end_time_seconds", end)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ObservationWindow":
        raw = _mapping(value, "window")
        try:
            start = raw["start_time_seconds"]
            end = raw["end_time_seconds"]
        except KeyError as error:
            raise ValueError(
                f"window is missing required field {error.args[0]!r}"
            ) from error
        return cls(
            start_time_seconds=start,
            end_time_seconds=end,
            start_inclusive=raw.get("start_inclusive", True),
            end_inclusive=raw.get("end_inclusive", True),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "start_time_seconds": self.start_time_seconds,
            "end_time_seconds": self.end_time_seconds,
            "start_inclusive": self.start_inclusive,
            "end_inclusive": self.end_inclusive,
        }


@dataclass(frozen=True, slots=True)
class EntityReference:
    entity_id: str
    role: EntityRole
    confidence: float = 1.0
    provenance: str = "source"

    def __post_init__(self) -> None:
        object.__setattr__(self, "entity_id", _text(self.entity_id, "entity_id"))
        object.__setattr__(self, "role", _enum_value(EntityRole, self.role, "role"))
        object.__setattr__(
            self, "confidence", _probability(self.confidence, "entity confidence")
        )
        object.__setattr__(self, "provenance", _text(self.provenance, "provenance"))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EntityReference":
        raw = _mapping(value, "entity reference")
        if "entity_id" not in raw or "role" not in raw:
            raise ValueError("entity reference requires entity_id and role")
        return cls(
            entity_id=raw["entity_id"],
            role=raw["role"],
            confidence=raw.get("confidence", 1.0),
            provenance=raw.get("provenance", "source"),
        )

    def to_agent_dict(self, policy: RedactionPolicy) -> dict[str, Any]:
        result: dict[str, Any] = {
            "entity_id": self.entity_id,
            "role": self.role.value,
            "confidence": self.confidence,
        }
        if policy.include_entity_provenance:
            result["provenance"] = self.provenance
        return result


@dataclass(frozen=True, slots=True)
class DataQuality:
    parse_confidence: float = 1.0
    missingness_fraction: float = 0.0
    delay_seconds: float = 0.0
    availability_mask: Mapping[str, bool] = field(
        default_factory=lambda: MappingProxyType({})
    )
    validation_flags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "parse_confidence",
            _probability(self.parse_confidence, "parse_confidence"),
        )
        object.__setattr__(
            self,
            "missingness_fraction",
            _probability(self.missingness_fraction, "missingness_fraction"),
        )
        delay = _finite_number(self.delay_seconds, "delay_seconds")
        if delay < 0.0:
            raise ValueError("delay_seconds must be non-negative")
        object.__setattr__(self, "delay_seconds", delay)

        raw_mask = _mapping(self.availability_mask, "availability_mask")
        mask: dict[str, bool] = {}
        for key, available in sorted(raw_mask.items()):
            key = _text(key, "availability mask key")
            if not isinstance(available, bool):
                raise TypeError("availability mask values must be booleans")
            mask[key] = available
        object.__setattr__(self, "availability_mask", MappingProxyType(mask))
        flags = tuple(_text(flag, "validation flag") for flag in self.validation_flags)
        object.__setattr__(self, "validation_flags", flags)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DataQuality":
        raw = _mapping(value, "data_quality")
        return cls(
            parse_confidence=raw.get("parse_confidence", 1.0),
            missingness_fraction=raw.get("missingness_fraction", 0.0),
            delay_seconds=raw.get("delay_seconds", 0.0),
            availability_mask=raw.get("availability_mask", {}),
            validation_flags=tuple(raw.get("validation_flags", ())),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "parse_confidence": self.parse_confidence,
            "missingness_fraction": self.missingness_fraction,
            "delay_seconds": self.delay_seconds,
            "availability_mask": dict(sorted(self.availability_mask.items())),
            "validation_flags": list(self.validation_flags),
        }


@dataclass(frozen=True, slots=True)
class ObservableMetadata:
    """Structural metadata available to both training and inference."""

    event_start_time_seconds: float
    event_end_time_seconds: float
    ingest_time_seconds: float | None
    available_at_time_seconds: float
    available_at_sequence: SequenceMarker | None
    entities: tuple[EntityReference, ...]
    primary_subsystem: str
    primary_subsystem_provenance: str
    correlation_ids: Mapping[str, str]
    source_references: tuple[str, ...]
    data_quality: DataQuality

    def __post_init__(self) -> None:
        start = _finite_number(
            self.event_start_time_seconds, "event_start_time_seconds"
        )
        end = _finite_number(self.event_end_time_seconds, "event_end_time_seconds")
        available = _finite_number(
            self.available_at_time_seconds, "available_at_time_seconds"
        )
        if start < 0.0 or end < start:
            raise ValueError("event interval must satisfy 0 <= start <= end")
        if available < 0.0:
            raise ValueError("available_at_time_seconds must be non-negative")
        ingest: float | None = None
        if self.ingest_time_seconds is not None:
            ingest = _finite_number(self.ingest_time_seconds, "ingest_time_seconds")
            if ingest < 0.0:
                raise ValueError("ingest_time_seconds must be non-negative")
            if ingest > available:
                raise ValueError(
                    "ingest_time_seconds must be <= available_at_time_seconds"
                )
        marker = self.available_at_sequence
        if marker is not None and (
            isinstance(marker, bool) or not isinstance(marker, (str, int))
        ):
            raise TypeError("available_at_sequence must be a string, integer, or None")
        if isinstance(marker, str):
            marker = _text(marker, "available_at_sequence")

        entities = tuple(self.entities)
        if not all(isinstance(entity, EntityReference) for entity in entities):
            raise TypeError("entities must contain EntityReference values")
        entity_keys = [(entity.entity_id, entity.role) for entity in entities]
        if len(set(entity_keys)) != len(entity_keys):
            raise ValueError("entity references must be unique by entity_id and role")

        subsystem = _text(self.primary_subsystem, "primary_subsystem")
        subsystem_provenance = _text(
            self.primary_subsystem_provenance,
            "primary_subsystem_provenance",
        )
        raw_correlations = _mapping(self.correlation_ids, "correlation_ids")
        correlations = {
            _text(key, "correlation identifier type"): _text(
                item, "correlation identifier"
            )
            for key, item in sorted(raw_correlations.items())
        }
        references = tuple(
            _text(reference, "source reference") for reference in self.source_references
        )
        if len(set(references)) != len(references):
            raise ValueError("source_references must not contain duplicates")
        if not isinstance(self.data_quality, DataQuality):
            raise TypeError("data_quality must be a DataQuality value")

        object.__setattr__(self, "event_start_time_seconds", start)
        object.__setattr__(self, "event_end_time_seconds", end)
        object.__setattr__(self, "ingest_time_seconds", ingest)
        object.__setattr__(self, "available_at_time_seconds", available)
        object.__setattr__(self, "available_at_sequence", marker)
        object.__setattr__(self, "entities", entities)
        object.__setattr__(self, "primary_subsystem", subsystem)
        object.__setattr__(self, "primary_subsystem_provenance", subsystem_provenance)
        object.__setattr__(self, "correlation_ids", MappingProxyType(correlations))
        object.__setattr__(self, "source_references", references)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ObservableMetadata":
        raw = _mapping(value, "metadata")
        required = {
            "event_start_time_seconds",
            "event_end_time_seconds",
            "available_at_time_seconds",
            "entities",
            "primary_subsystem",
            "data_quality",
        }
        missing = sorted(required - set(raw))
        if missing:
            raise ValueError(f"metadata is missing required fields: {missing}")
        entities_raw = raw["entities"]
        if not isinstance(entities_raw, Sequence) or isinstance(entities_raw, str):
            raise TypeError("metadata.entities must be an array")
        return cls(
            event_start_time_seconds=raw["event_start_time_seconds"],
            event_end_time_seconds=raw["event_end_time_seconds"],
            ingest_time_seconds=raw.get("ingest_time_seconds"),
            available_at_time_seconds=raw["available_at_time_seconds"],
            available_at_sequence=raw.get("available_at_sequence"),
            entities=tuple(EntityReference.from_dict(item) for item in entities_raw),
            primary_subsystem=raw["primary_subsystem"],
            primary_subsystem_provenance=raw.get(
                "primary_subsystem_provenance", "source"
            ),
            correlation_ids=raw.get("correlation_ids", {}),
            source_references=tuple(raw.get("source_references", ())),
            data_quality=DataQuality.from_dict(raw["data_quality"]),
        )

    def validate_available(
        self,
        query_time_seconds: float,
        query_watermark_sequence: SequenceMarker | None = None,
    ) -> None:
        query_time = _finite_number(query_time_seconds, "query_time_seconds")
        if self.event_end_time_seconds > query_time:
            raise ValueError("observation contains future event telemetry")
        if self.available_at_time_seconds > query_time:
            raise ValueError("observation was not available at query time")
        if (
            self.ingest_time_seconds is not None
            and self.ingest_time_seconds > query_time
        ):
            raise ValueError("observation was ingested after query time")
        if query_watermark_sequence is None or self.available_at_sequence is None:
            return
        available = self.available_at_sequence
        query = query_watermark_sequence
        if isinstance(available, int) and isinstance(query, int):
            if available > query:
                raise ValueError(
                    "observation sequence is newer than the query watermark"
                )
        elif available != query:
            raise ValueError(
                "opaque available_at_sequence must match query_watermark_sequence"
            )

    def to_agent_dict(self, policy: RedactionPolicy) -> dict[str, Any]:
        result: dict[str, Any] = {
            "event_start_time_seconds": self.event_start_time_seconds,
            "event_end_time_seconds": self.event_end_time_seconds,
            "available_at_time_seconds": self.available_at_time_seconds,
        }
        if self.ingest_time_seconds is not None:
            result["ingest_time_seconds"] = self.ingest_time_seconds
        if policy.include_entities:
            allowed_roles = set(policy.allowed_entity_roles)
            result["entities"] = [
                entity.to_agent_dict(policy)
                for entity in self.entities
                if entity.role in allowed_roles
            ]
        if policy.include_primary_subsystem:
            result["primary_subsystem"] = self.primary_subsystem
            if policy.include_subsystem_provenance:
                result["primary_subsystem_provenance"] = (
                    self.primary_subsystem_provenance
                )
        if policy.include_correlation_ids:
            result["correlation_ids"] = dict(sorted(self.correlation_ids.items()))
        if policy.include_source_references:
            result["source_references"] = list(self.source_references)
        if policy.include_data_quality:
            result["data_quality"] = self.data_quality.to_dict()
        return result


@dataclass(frozen=True, slots=True)
class CanonicalObservation:
    """One inference-safe canonical observation :math:`o_{I,i}`."""

    observation_id: str
    channel: TelemetryChannel
    window: ObservationWindow
    payload: Mapping[str, JSONValue]
    metadata: ObservableMetadata

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "observation_id", _text(self.observation_id, "observation_id")
        )
        object.__setattr__(
            self, "channel", _enum_value(TelemetryChannel, self.channel, "channel")
        )
        if not isinstance(self.window, ObservationWindow):
            raise TypeError("window must be an ObservationWindow")
        if not isinstance(self.metadata, ObservableMetadata):
            raise TypeError("metadata must be ObservableMetadata")
        payload = _mapping(self.payload, "payload")
        _reject_hidden_keys(payload, "payload")
        object.__setattr__(self, "payload", _freeze_json(payload, "payload"))
        if not math.isclose(
            self.window.start_time_seconds,
            self.metadata.event_start_time_seconds,
            rel_tol=0.0,
            abs_tol=1e-6,
        ) or not math.isclose(
            self.window.end_time_seconds,
            self.metadata.event_end_time_seconds,
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            raise ValueError("observation window must match metadata event interval")

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, Any],
        *,
        query_time_seconds: float | None = None,
        query_watermark_sequence: SequenceMarker | None = None,
    ) -> "CanonicalObservation":
        raw = _mapping(value, "observation")
        _reject_hidden_keys(raw)
        required = {"observation_id", "channel", "window", "payload", "metadata"}
        missing = sorted(required - set(raw))
        if missing:
            raise ValueError(f"observation is missing required fields: {missing}")
        observation = cls(
            observation_id=raw["observation_id"],
            channel=raw["channel"],
            window=ObservationWindow.from_dict(raw["window"]),
            payload=raw["payload"],
            metadata=ObservableMetadata.from_dict(raw["metadata"]),
        )
        if query_time_seconds is not None:
            observation.validate_available(query_time_seconds, query_watermark_sequence)
        return observation

    def validate_available(
        self,
        query_time_seconds: float,
        query_watermark_sequence: SequenceMarker | None = None,
    ) -> None:
        query_time = _finite_number(query_time_seconds, "query_time_seconds")
        if self.window.end_time_seconds > query_time:
            raise ValueError("observation window extends beyond query time")
        self.metadata.validate_available(query_time, query_watermark_sequence)

    def to_agent_dict(self, policy: RedactionPolicy | None = None) -> dict[str, Any]:
        policy = policy or RedactionPolicy()
        result: dict[str, Any] = {
            "observation_id": self.observation_id,
            "channel": self.channel.value,
            "window": self.window.to_dict(),
            "metadata": self.metadata.to_agent_dict(policy),
        }
        if policy.include_payload:
            payload = _thaw_json(self.payload)
            allowlist = policy.payload_allowlist.get(self.channel)
            allowed = set(allowlist) if allowlist else None
            denied = set(policy.payload_denylist)
            result["payload"] = {
                key: payload[key]
                for key in sorted(payload)
                if key not in denied and (allowed is None or key in allowed)
            }
        return result


@dataclass(frozen=True, slots=True)
class CanonicalIncident:
    """All canonical observations causally available for one query."""

    incident_id: str
    query_time_seconds: float
    observations: tuple[CanonicalObservation, ...]
    snapshot_id: str | None = None
    query_watermark_sequence: SequenceMarker | None = None
    schema_version: str = CANONICAL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        incident_id = _text(self.incident_id, "incident_id")
        query_time = _finite_number(self.query_time_seconds, "query_time_seconds")
        if query_time < 0.0:
            raise ValueError("query_time_seconds must be non-negative")
        observations = tuple(self.observations)
        if not observations:
            raise ValueError("an incident must contain at least one observation")
        if not all(isinstance(item, CanonicalObservation) for item in observations):
            raise TypeError("observations must contain CanonicalObservation values")
        observation_ids = [item.observation_id for item in observations]
        if len(set(observation_ids)) != len(observation_ids):
            raise ValueError(
                "observation identifiers must be unique within an incident"
            )
        snapshot_id = self.snapshot_id
        if snapshot_id is not None:
            snapshot_id = _text(snapshot_id, "snapshot_id")
        marker = self.query_watermark_sequence
        if marker is not None and (
            isinstance(marker, bool) or not isinstance(marker, (str, int))
        ):
            raise TypeError(
                "query_watermark_sequence must be a string, integer, or None"
            )
        if isinstance(marker, str):
            marker = _text(marker, "query_watermark_sequence")
        schema = _text(self.schema_version, "schema_version")
        if schema != CANONICAL_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported canonical schema {schema!r}; "
                f"expected {CANONICAL_SCHEMA_VERSION!r}"
            )
        for observation in observations:
            observation.validate_available(query_time, marker)

        object.__setattr__(self, "incident_id", incident_id)
        object.__setattr__(self, "query_time_seconds", query_time)
        object.__setattr__(self, "observations", observations)
        object.__setattr__(self, "snapshot_id", snapshot_id)
        object.__setattr__(self, "query_watermark_sequence", marker)
        object.__setattr__(self, "schema_version", schema)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CanonicalIncident":
        """Validate a canonical snapshot without accepting a label sidecar."""

        raw = _mapping(value, "canonical snapshot")
        _reject_hidden_keys(raw, "snapshot")
        incident_id = raw.get("incident_id", raw.get("episode_id"))
        if incident_id is None:
            raise ValueError("canonical snapshot requires incident_id or episode_id")
        if "query_time_seconds" not in raw:
            raise ValueError("canonical snapshot requires query_time_seconds")
        observations_raw = raw.get("observations")
        if not isinstance(observations_raw, Sequence) or isinstance(
            observations_raw, (str, bytes, bytearray)
        ):
            raise TypeError("canonical snapshot observations must be an array")
        query_time = _finite_number(raw["query_time_seconds"], "query_time_seconds")
        marker = raw.get("query_watermark_sequence")
        observations = tuple(
            (
                item
                if isinstance(item, CanonicalObservation)
                else CanonicalObservation.from_dict(
                    item,
                    query_time_seconds=query_time,
                    query_watermark_sequence=marker,
                )
            )
            for item in observations_raw
        )
        return cls(
            incident_id=incident_id,
            query_time_seconds=query_time,
            observations=observations,
            snapshot_id=raw.get("snapshot_id"),
            query_watermark_sequence=marker,
            schema_version=raw.get("schema_version", CANONICAL_SCHEMA_VERSION),
        )

    def observation_by_id(self, observation_id: str) -> CanonicalObservation:
        for observation in self.observations:
            if observation.observation_id == observation_id:
                return observation
        raise KeyError(observation_id)


@dataclass(frozen=True, slots=True)
class IncidentBatch:
    """Observable-only model input for one or more causal snapshot rows.

    Each row has a unique snapshot-qualified ``(incident_id, snapshot_id)``
    key so separate causal cuts retain distinct identities.
    """

    incidents: tuple[CanonicalIncident, ...]

    def __post_init__(self) -> None:
        incidents = tuple(self.incidents)
        if not incidents:
            raise ValueError("IncidentBatch must contain at least one incident")
        if not all(isinstance(item, CanonicalIncident) for item in incidents):
            raise TypeError("incidents must contain CanonicalIncident values")
        row_keys = tuple((item.incident_id, item.snapshot_id) for item in incidents)
        if len(set(row_keys)) != len(row_keys):
            raise ValueError(
                "snapshot-qualified incident identifiers must be unique within a batch"
            )
        incident_counts: dict[str, int] = {}
        for incident in incidents:
            incident_counts[incident.incident_id] = (
                incident_counts.get(incident.incident_id, 0) + 1
            )
        missing_snapshot_ids = sorted(
            incident.incident_id
            for incident in incidents
            if incident_counts[incident.incident_id] > 1
            and incident.snapshot_id is None
        )
        if missing_snapshot_ids:
            raise ValueError(
                "repeated incident identifiers require snapshot_id values: "
                f"{missing_snapshot_ids}"
            )
        object.__setattr__(self, "incidents", incidents)

    @classmethod
    def from_snapshots(
        cls, snapshots: Sequence[Mapping[str, Any] | CanonicalIncident]
    ) -> "IncidentBatch":
        return cls(
            incidents=tuple(
                (
                    snapshot
                    if isinstance(snapshot, CanonicalIncident)
                    else CanonicalIncident.from_dict(snapshot)
                )
                for snapshot in snapshots
            )
        )

    @property
    def incident_ids(self) -> tuple[str, ...]:
        return tuple(incident.incident_id for incident in self.incidents)

    @property
    def snapshot_keys(self) -> tuple[tuple[str, str | None], ...]:
        """Return stable causal-row identifiers without exposing supervision."""

        return tuple(
            (incident.incident_id, incident.snapshot_id) for incident in self.incidents
        )


@dataclass(frozen=True, slots=True)
class RootCauseClass:
    """One learned root-cause prototype class."""

    class_id: str
    mechanism: str
    target_type: str
    description: str
    prototype_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "class_id", _text(self.class_id, "class_id"))
        object.__setattr__(self, "mechanism", _text(self.mechanism, "mechanism"))
        object.__setattr__(self, "target_type", _text(self.target_type, "target_type"))
        object.__setattr__(self, "description", _text(self.description, "description"))
        if self.prototype_id is not None:
            object.__setattr__(
                self, "prototype_id", _text(self.prototype_id, "prototype_id")
            )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RootCauseClass":
        raw = _mapping(value, "root-cause class")
        prototype_id = raw.get("prototype_id")
        class_id = raw.get("class_id", prototype_id)
        if class_id is None:
            raise ValueError("root-cause class requires class_id or prototype_id")
        return cls(
            class_id=class_id,
            mechanism=raw["mechanism"],
            target_type=raw.get("target_type", "unknown"),
            description=raw.get("description", raw.get("operational_description", "")),
            prototype_id=prototype_id,
        )


@dataclass(frozen=True, slots=True)
class RootCauseCatalog:
    """Ordered class catalog defining root-cause logit indices."""

    classes: tuple[RootCauseClass, ...]

    def __post_init__(self) -> None:
        classes = tuple(self.classes)
        if not classes:
            raise ValueError("root-cause catalog must contain at least one class")
        if not all(isinstance(item, RootCauseClass) for item in classes):
            raise TypeError("classes must contain RootCauseClass values")
        class_ids = [item.class_id for item in classes]
        if len(set(class_ids)) != len(class_ids):
            raise ValueError("root-cause class identifiers must be unique")
        object.__setattr__(self, "classes", classes)

    @classmethod
    def from_dicts(cls, values: Sequence[Mapping[str, Any]]) -> "RootCauseCatalog":
        return cls(tuple(RootCauseClass.from_dict(item) for item in values))

    @property
    def class_ids(self) -> tuple[str, ...]:
        return tuple(item.class_id for item in self.classes)

    def index(self, class_id: str) -> int:
        try:
            return self.class_ids.index(class_id)
        except ValueError as error:
            raise KeyError(class_id) from error

    def get(self, class_id: str) -> RootCauseClass:
        return self.classes[self.index(class_id)]


@dataclass(frozen=True, slots=True)
class ModelOutput:
    """The learned aspect embeddings and relevance scores used at inference."""

    aspect_embeddings: Tensor
    relevance_scores: Tensor

    def __post_init__(self):
        if self.aspect_embeddings.ndim != 3 or self.relevance_scores.ndim != 2:
            raise ValueError("invalid inference tensor rank")
        if self.aspect_embeddings.shape[:2] != self.relevance_scores.shape:
            raise ValueError("aspect/relevance cardinality mismatch")


@dataclass(frozen=True, slots=True)
class SelectedAnchor:
    """Internal result of budgeted quality-diversity anchor selection.

    ``token_cost`` is the candidate's singleton cost estimate ``ell_i``. Exact
    B enforcement uses the jointly serialized bundle cost, whose shared table
    and group envelopes are intentionally not allocated back to one anchor.
    """

    observation: CanonicalObservation
    observation_index: int
    relevance_score: float
    token_cost: int

    def __post_init__(self) -> None:
        if not isinstance(self.observation, CanonicalObservation):
            raise TypeError("observation must be a CanonicalObservation")
        if (
            isinstance(self.observation_index, bool)
            or not isinstance(self.observation_index, int)
            or self.observation_index < 0
        ):
            raise ValueError("observation_index must be a non-negative integer")
        if (
            isinstance(self.token_cost, bool)
            or not isinstance(self.token_cost, int)
            or self.token_cost < 1
        ):
            raise ValueError("token_cost must be a positive integer")
        object.__setattr__(
            self,
            "relevance_score",
            _finite_number(self.relevance_score, "relevance_score"),
        )


@dataclass(frozen=True, slots=True)
class CorroboratingEvidenceGroup:
    """One selected anchor and its non-anchor corroborating observations.

    ``token_cost`` is the sum of member singleton estimates for inspection; it
    is not additive across groups. ``StateBundleOutput.used_tokens`` is the
    authoritative jointly packed bundle cost.
    """

    anchor: SelectedAnchor
    evidence: tuple[CanonicalObservation, ...]
    token_cost: int

    def __post_init__(self) -> None:
        if not isinstance(self.anchor, SelectedAnchor):
            raise TypeError("anchor must be a SelectedAnchor")
        observations = tuple(self.evidence)
        if not all(isinstance(item, CanonicalObservation) for item in observations):
            raise TypeError("evidence must contain CanonicalObservation values")
        identifiers = [item.observation_id for item in observations]
        if self.anchor.observation.observation_id in identifiers:
            raise ValueError("an anchor cannot corroborate itself")
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("corroborating observations must be unique within a group")
        if (
            isinstance(self.token_cost, bool)
            or not isinstance(self.token_cost, int)
            or self.token_cost < self.anchor.token_cost
        ):
            raise ValueError("group token_cost must include the anchor token cost")
        object.__setattr__(self, "evidence", observations)


@dataclass(frozen=True, slots=True)
class StateBundleOutput:
    """Final grouped, agent-safe StateBundle result.

    ``to_dict`` and ``to_json`` intentionally omit selector scores,
    per-observation token costs, soft gates, prototype distributions, and dense
    embeddings.  ``selection_audit`` is retained only by the trusted runtime;
    it never crosses the agent boundary. Aggregate budget usage and a safe
    targeted-request status remain visible for auditability. Target-scope
    candidates are explicitly selector estimates derived from observable
    evidence, not training/evaluator causal labels.
    """

    incident_id: str
    query_time_seconds: float
    groups: tuple[CorroboratingEvidenceGroup, ...]
    token_budget: int
    used_tokens: int
    candidate_count: int
    snapshot_id: str | None = None
    redaction_policy: RedactionPolicy = field(default_factory=RedactionPolicy)
    request_status: Mapping[str, JSONValue] = field(
        default_factory=lambda: MappingProxyType({})
    )
    target_scope_ambiguity: bool = False
    target_scope_candidates: tuple[Mapping[str, JSONValue], ...] = ()
    selection_audit: Mapping[str, JSONValue] = field(
        default_factory=lambda: MappingProxyType({})
    )
    schema_version: str = STATEBUNDLE_OUTPUT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "incident_id", _text(self.incident_id, "incident_id"))
        query_time = _finite_number(self.query_time_seconds, "query_time_seconds")
        if query_time < 0.0:
            raise ValueError("query_time_seconds must be non-negative")
        groups = tuple(self.groups)
        if not all(isinstance(item, CorroboratingEvidenceGroup) for item in groups):
            raise TypeError("groups must contain CorroboratingEvidenceGroup values")
        indices = [group.anchor.observation_index for group in groups]
        anchor_ids = [group.anchor.observation.observation_id for group in groups]
        if len(set(indices)) != len(indices):
            raise ValueError("anchor observation indices must be unique")
        if len(set(anchor_ids)) != len(anchor_ids):
            raise ValueError("anchor observations must be unique")
        selected_observation_ids = frozenset(
            observation.observation_id
            for group in groups
            for observation in (group.anchor.observation, *group.evidence)
        )
        for name in ("token_budget", "used_tokens", "candidate_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.token_budget < 1:
            raise ValueError("token_budget must be positive")
        if self.used_tokens > self.token_budget:
            raise ValueError("used_tokens must not exceed token_budget")
        if self.candidate_count < len(groups):
            raise ValueError("candidate_count must be at least the anchor count")
        snapshot_id = self.snapshot_id
        if snapshot_id is not None:
            snapshot_id = _text(snapshot_id, "snapshot_id")
        if not isinstance(self.redaction_policy, RedactionPolicy):
            raise TypeError("redaction_policy must be a RedactionPolicy")
        if not isinstance(self.target_scope_ambiguity, bool):
            raise TypeError("target_scope_ambiguity must be a boolean")
        target_scope_candidates = _target_scope_candidates(
            self.target_scope_candidates,
            selected_observation_ids=selected_observation_ids,
        )
        if self.target_scope_ambiguity and len(target_scope_candidates) < 2:
            raise ValueError(
                "target_scope_ambiguity requires at least two target scopes"
            )
        request_status = _mapping(self.request_status, "request_status")
        selection_audit = _mapping(self.selection_audit, "selection_audit")
        _reject_hidden_keys(request_status, "request_status")
        schema = _text(self.schema_version, "schema_version")
        if schema != STATEBUNDLE_OUTPUT_SCHEMA_VERSION:
            raise ValueError(f"unsupported StateBundle output schema {schema!r}")
        object.__setattr__(self, "query_time_seconds", query_time)
        object.__setattr__(self, "groups", groups)
        object.__setattr__(self, "snapshot_id", snapshot_id)
        object.__setattr__(
            self,
            "target_scope_candidates",
            target_scope_candidates,
        )
        object.__setattr__(
            self,
            "request_status",
            _freeze_json(request_status, "request_status"),
        )
        object.__setattr__(
            self,
            "selection_audit",
            _freeze_json(selection_audit, "selection_audit"),
        )
        object.__setattr__(self, "schema_version", schema)

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic agent-visible data with internal fields removed."""

        # Inference already emits groups in deterministic priority order.
        # Preserve that order so protected/fallback evidence is not pushed
        # below lower-priority rows merely because it appeared later in the
        # canonical snapshot.
        ordered_groups = self.groups
        result: dict[str, Any] = {
            "schema_version": self.schema_version,
            "incident_id": self.incident_id,
            "query_time_seconds": self.query_time_seconds,
            "token_budget": self.token_budget,
            "used_tokens": self.used_tokens,
            "candidate_count": self.candidate_count,
            "target_scope_ambiguity": self.target_scope_ambiguity,
            "target_scope_candidates": [
                _thaw_json(candidate) for candidate in self.target_scope_candidates
            ],
            "evidence_groups": [
                {
                    "anchor": group.anchor.observation.to_agent_dict(
                        self.redaction_policy
                    ),
                    "corroborating_observations": [
                        observation.to_agent_dict(self.redaction_policy)
                        for observation in group.evidence
                    ],
                }
                for group in ordered_groups
            ],
        }
        if self.snapshot_id is not None:
            result["snapshot_id"] = self.snapshot_id
        if self.request_status:
            result["request_status"] = _thaw_json(self.request_status)
        return result

    def audit_dict(self) -> dict[str, Any]:
        """Return trusted selector diagnostics excluded from agent output."""

        return _thaw_json(self.selection_audit)

    def to_json(self, *, indent: int | None = None) -> str:
        """Serialize agent-visible output with stable key and group ordering."""

        separators = (",", ":") if indent is None else None
        return json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=separators,
            indent=indent,
            ensure_ascii=False,
            allow_nan=False,
        )


__all__ = [
    "CANONICAL_SCHEMA_VERSION",
    "STATEBUNDLE_OUTPUT_SCHEMA_VERSION",
    "CanonicalIncident",
    "CanonicalObservation",
    "CorroboratingEvidenceGroup",
    "DataQuality",
    "EntityReference",
    "EntityRole",
    "IncidentBatch",
    "ModelOutput",
    "ObservableMetadata",
    "ObservationWindow",
    "RedactionPolicy",
    "RootCauseCatalog",
    "RootCauseClass",
    "SelectedAnchor",
    "StateBundleOutput",
    "TelemetryChannel",
    "assert_observable_only",
]
