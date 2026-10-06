"""Observable-only deterministic safeguards for StateBundle inference.

This module deliberately has no model, checkpoint, simulator, evaluator, or
training-label dependency.  It annotates canonical observations using fields
that are present at the causal cut so the learned relevance score can be
reranked and packed more safely at inference time.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
import hashlib
import json
import math
import re
from typing import Any

from aiopslab.agent_telemetry import AgentObservationRequest, filter_observations
from aiopslab.statebundle.config import StateBundleInferenceConfig
from aiopslab.statebundle.types import (
    CanonicalObservation,
    EntityRole,
    TelemetryChannel,
)


_GENERIC_ENTITY_IDS = frozenset(
    {
        "application",
        "datacenter",
        "global",
        "monitoring",
        "network",
        "storage",
        "unknown",
        "workload",
    }
)
_NON_SPECIFIC_TARGET_IDS = frozenset({"datacenter", "global", "unknown"})
_DEFAULT_REQUEST = AgentObservationRequest()
_SLO_TERMS = (
    "health_violation",
    "healthconstraint",
    "health_constraint",
    "sla_violation",
    "sloviolation",
    "slo_violation",
    "thermal_critical",
    "budget_violation",
    "placement_policy_violations",
    "failed_server_count",
    "health_flapping_count",
    "sensor_unhealthy",
    "unhealthy_count",
)
_BOOKKEEPING_TERMS = (
    "auto_advance",
    "heartbeat",
    "noop",
    "sim_advance",
    "sim_reset",
    "sim_step",
    "simulation_advance",
    "snapshot",
    "telemetry_capture",
    "workload_started",
)
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_LINK_STRENGTH_PRIORITY = {
    "no_match": 0,
    "weak_context_match": 1,
    "entity_and_symptom_match": 2,
    "direct_counterpart": 3,
}
_MEASURE_STOP_TERMS = frozenset(
    {
        "average",
        "current",
        "count",
        "gauge",
        "maximum",
        "mean",
        "minimum",
        "ms",
        "number",
        "percent",
        "per",
        "ratio",
        "second",
        "seconds",
        "total",
        "value",
    }
)
_ENTITY_SCOPE_TERMS = frozenset(
    {
        "application",
        "control",
        "controlplane",
        "datacenter",
        "network",
        "rack",
        "scheduler",
        "server",
        "service",
        "storage",
        "tenant",
        "thermal",
        "tor",
        "workload",
    }
)
_LOCAL_COMPONENT_TERMS = frozenset(
    {
        "autoscaler",
        "control",
        "controller",
        "cooling",
        "fan",
        "loadbalancer",
        "monitoring",
        "network",
        "placement",
        "rack",
        "scheduler",
        "sensor",
        "server",
        "storage",
        "thermal",
        "tor",
    }
)
_IMPACT_ALERT_TERMS = frozenset(
    {
        "capacitydrop",
        "latencystep",
        "servicecapacity",
        "serviceimpact",
        "sla",
        "trafficdrop",
    }
)


@dataclass(frozen=True, slots=True)
class ObservableSignals:
    """Auditable inference features derived from one canonical observation."""

    logical_key: str
    semantic_group: str
    entity_ids: tuple[str, ...]
    target_ids: tuple[str, ...]
    producer_ids: tuple[str, ...] = ()
    affected_ids: tuple[str, ...] = ()
    source_ids: tuple[str, ...] = ()
    destination_ids: tuple[str, ...] = ()
    scope_ids: tuple[str, ...] = ()
    metric_name: str | None = None
    alert_name: str | None = None
    severity: str | None = None
    active_alert: bool = False
    target_bearing: bool = False
    concrete_entity: bool = False
    slo_violation: bool = False
    linked_to_active_alert: bool = False
    alert_linkage_strength: str = "no_match"
    alert_linkage_reason: str | None = None
    matched_alert_field: str | None = None
    matched_alert_entity: str | None = None
    matched_alert_observation_id: str | None = None
    equivalence_group_id: str = ""
    estimated_target_role: str = "contextual_peer"
    estimated_scope_ids: tuple[str, ...] = ()
    anomalous_metric: bool = False
    zero_series: bool = False
    constant_series: bool = False
    normal_comparison: bool = False
    recent_config_change: bool = False
    static_config: bool = False
    stale_config: bool = False
    bookkeeping_log: bool = False
    request_eligible: bool = True
    request_match: bool = False
    protection_reasons: tuple[str, ...] = ()


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_thaw(item) for item in value]
    return value


def _stable_digest(value: Any, *, length: int = 20) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:length]


def _normalized_text(value: Any) -> str:
    if isinstance(value, Mapping):
        return " ".join(
            f"{_normalized_text(key)} {_normalized_text(item)}"
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return " ".join(_normalized_text(item) for item in value)
    return " ".join(_TOKEN_RE.findall(str(value).lower()))


def _normalized_measure_terms(value: Any) -> set[str]:
    """Return conservative observable measurement terms.

    Singularization and a few spelling aliases make canonical alert fields
    such as ``dropped_requests_per_second`` comparable to metric identities
    such as ``workload.dropped_requests``.  Entity/scope words remain
    available to the caller and are not themselves sufficient for a link.
    """

    aliases = {
        "dropped": "drop",
        "dropping": "drop",
        "errors": "error",
        "failed": "failure",
        "failures": "failure",
        "latencies": "latency",
        "operations": "operation",
        "packets": "packet",
        "requests": "request",
        "retransmits": "retransmit",
        "violations": "violation",
    }
    return {
        aliases.get(term, term)
        for term in _TOKEN_RE.findall(_normalized_text(value))
        if aliases.get(term, term) not in _MEASURE_STOP_TERMS
    }


def _metric_measure_terms(metric_name: str) -> tuple[set[str], set[str]]:
    terms = _normalized_measure_terms(metric_name)
    scope_hints = terms & _ENTITY_SCOPE_TERMS
    return terms - _ENTITY_SCOPE_TERMS, scope_hints


def _flatten_alert_fields(value: Any, *, prefix: str = "") -> list[tuple[str, Any]]:
    """Flatten only diagnostic alert details/threshold fields for matching."""

    if not isinstance(value, Mapping):
        return []
    rows: list[tuple[str, Any]] = []
    for raw_key, item in sorted(value.items(), key=lambda pair: str(pair[0])):
        key = str(raw_key)
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(item, Mapping):
            rows.extend(_flatten_alert_fields(item, prefix=path))
        elif not (
            isinstance(item, Sequence)
            and not isinstance(item, (str, bytes, bytearray))
        ):
            rows.append((path, item))
    return rows


def _windows_compatible(
    left: CanonicalObservation,
    right: CanonicalObservation,
    *,
    max_gap_seconds: float,
) -> bool:
    left_start = left.metadata.event_start_time_seconds
    left_end = left.metadata.event_end_time_seconds
    right_start = right.metadata.event_start_time_seconds
    right_end = right.metadata.event_end_time_seconds
    gap = max(left_start, right_start) - min(left_end, right_end)
    return gap <= max_gap_seconds


def _numeric_counterpart(
    alert_value: Any,
    metric_value: float | None,
    *,
    relative_tolerance: float,
    epsilon: float,
) -> bool:
    if (
        isinstance(alert_value, bool)
        or not isinstance(alert_value, (int, float))
        or metric_value is None
    ):
        return False
    alert_number = float(alert_value)
    if not math.isfinite(alert_number):
        return False
    scale = max(abs(alert_number), abs(metric_value), epsilon)
    return abs(alert_number - metric_value) <= max(
        epsilon, relative_tolerance * scale
    )


def _scope_hint_matches_metric(
    field_hints: set[str], metric_hints: set[str], entity_ids: Sequence[str]
) -> bool:
    if field_hints & metric_hints:
        return True
    normalized_entities = {
        term
        for entity in entity_ids
        for term in _TOKEN_RE.findall(entity.lower().replace("-", "_"))
    }
    return bool(field_hints & normalized_entities)


def _component_local_alert(
    observation: CanonicalObservation, signal: ObservableSignals
) -> bool:
    terms = set(
        _TOKEN_RE.findall(
            _normalized_text(
                {
                    "type": signal.alert_name,
                    "target": observation.payload.get("target"),
                    "subsystem": observation.metadata.primary_subsystem,
                }
            ).replace(" ", "")
        )
    )
    compact = "".join(sorted(terms))
    if any(term in compact for term in _IMPACT_ALERT_TERMS):
        return False
    target_terms = set(
        _TOKEN_RE.findall(str(observation.payload.get("target", "")).lower())
    )
    subsystem_terms = set(
        _TOKEN_RE.findall(observation.metadata.primary_subsystem.lower())
    )
    alert_terms = set(
        _TOKEN_RE.findall(_normalized_text(signal.alert_name or ""))
    )
    return bool(
        alert_terms & _LOCAL_COMPONENT_TERMS
        or alert_terms & target_terms
        or alert_terms & subsystem_terms
        or (
            observation.metadata.primary_subsystem.lower()
            not in {"application", "operations", "unknown", "workload"}
            and signal.concrete_entity
        )
        or "applicationerror" in compact
    )


def _scope_estimates(
    observation: CanonicalObservation,
    signal: ObservableSignals,
    *,
    role: str,
) -> tuple[str, ...]:
    subsystem = observation.metadata.primary_subsystem.lower().replace("-", "_")
    entities = [*signal.target_ids, *signal.entity_ids]
    if any(item.lower() == "workload" for item in entities):
        return ("workload",)
    if subsystem in {"control_plane", "controlplane"}:
        return ("control_plane",)
    if role == "direct_target_candidate" and subsystem not in {
        "application",
        "configuration",
        "operations",
        "unknown",
        "workload",
    }:
        return (subsystem,)
    concrete = sorted(
        {
            item
            for item in (*signal.target_ids, *signal.entity_ids)
            if item.lower() not in _GENERIC_ENTITY_IDS
            and item.lower() not in {"default", "unknown"}
        }
    )
    if concrete:
        return tuple(concrete)
    if subsystem == "application":
        return ("application",)
    return tuple(
        sorted(
            {
                item
                for item in (*signal.target_ids, *signal.scope_ids)
                if item.lower() not in _NON_SPECIFIC_TARGET_IDS
            }
        )
    )


def _value_shape_class(
    signal: ObservableSignals, observation: CanonicalObservation
) -> str:
    if signal.zero_series:
        return "near_zero_constant"
    if signal.constant_series:
        return "constant"
    if signal.anomalous_metric:
        return "changing_or_anomalous"
    if observation.channel is TelemetryChannel.ALERT:
        return "active" if signal.active_alert else "inactive"
    if observation.channel is TelemetryChannel.CONFIG:
        return "recent_change" if signal.recent_config_change else "static"
    return "ordinary"


def _equivalence_group_id(
    observation: CanonicalObservation, signal: ObservableSignals
) -> str:
    """Modality-aware observable equivalence, never a hidden-label grouping."""

    payload = _thaw(observation.payload)
    role = signal.estimated_target_role
    state = _value_shape_class(signal, observation)
    subsystem = observation.metadata.primary_subsystem
    if observation.channel is TelemetryChannel.METRIC:
        # Preserve each abnormal/direct target independently, while allowing
        # healthy fleet peers with the same shape to share representatives.
        metric_domain = str(signal.metric_name or "unknown").split(".", 1)[0]
        entity_scope: Any = (
            signal.entity_ids
            if role == "direct_target_candidate" and signal.anomalous_metric
            else (
                "producer_peer",
                metric_domain,
            )
        )
        key: Any = (
            "metric",
            signal.metric_name,
            subsystem,
            role,
            state,
            signal.semantic_group,
            entity_scope,
        )
    elif observation.channel is TelemetryChannel.ALERT:
        key = (
            "alert",
            signal.alert_name,
            str(payload.get("status", "unknown")).lower(),
            str(payload.get("target", "unknown")),
            subsystem,
            role,
        )
    elif observation.channel is TelemetryChannel.CONFIG:
        key = (
            "config",
            payload.get("scope"),
            payload.get("path"),
            payload.get("value"),
            state,
        )
    elif observation.channel is TelemetryChannel.LOG:
        key = (
            "log",
            payload.get("template_id", payload.get("event_type")),
            subsystem,
            role,
            state,
        )
    else:
        key = (
            "trace",
            payload.get("operation"),
            payload.get("source"),
            payload.get("destination"),
            payload.get("status_counts"),
            role,
        )
    return f"eq:{observation.channel.value}:{_stable_digest(key)}"


def _numeric_values(payload: Mapping[str, Any]) -> tuple[float, ...]:
    raw_values = payload.get("values")
    if not isinstance(raw_values, Sequence) or isinstance(raw_values, (str, bytes)):
        return ()
    missing = payload.get("missingness_mask")
    masks = (
        tuple(bool(item) for item in missing)
        if isinstance(missing, Sequence) and not isinstance(missing, (str, bytes))
        else ()
    )
    values: list[float] = []
    for index, value in enumerate(raw_values):
        if index < len(masks) and masks[index]:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        numeric = float(value)
        if math.isfinite(numeric):
            values.append(numeric)
    return tuple(values)


def _last_metric_value(
    payload: Mapping[str, Any], values: Sequence[float]
) -> float | None:
    statistics = payload.get("statistics")
    if isinstance(statistics, Mapping):
        value = statistics.get("last")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            numeric = float(value)
            if math.isfinite(numeric):
                return numeric
    return float(values[-1]) if values else None


def _metric_anomalous(
    payload: Mapping[str, Any],
    values: Sequence[float],
    config: StateBundleInferenceConfig,
) -> bool:
    if not values:
        return False
    last = _last_metric_value(payload, values)
    reference = payload.get("normalization_reference")
    if last is not None and isinstance(reference, Mapping):
        median = reference.get("median")
        iqr = reference.get("iqr")
        if (
            isinstance(median, (int, float))
            and not isinstance(median, bool)
            and isinstance(iqr, (int, float))
            and not isinstance(iqr, bool)
        ):
            median_value = float(median)
            iqr_value = abs(float(iqr))
            delta = abs(last - median_value)
            if iqr_value > config.zero_absolute_epsilon:
                if delta / iqr_value >= config.metric_anomaly_iqr_threshold:
                    return True
            else:
                threshold = max(
                    config.zero_absolute_epsilon,
                    abs(median_value) * config.metric_change_fraction_threshold,
                )
                if delta > threshold:
                    return True
    first = float(values[0])
    last_value = float(values[-1])
    threshold = max(
        config.zero_absolute_epsilon,
        abs(first) * config.metric_change_fraction_threshold,
    )
    return abs(last_value - first) > threshold


def _metric_semantic_group(
    metric_name: str,
    values: Sequence[float],
    *,
    zero_series: bool,
    round_digits: int,
) -> str:
    if zero_series:
        return f"metric-zero:{metric_name}"
    if values:
        scale = max(1.0, max(abs(item) for item in values))
        normalized = tuple(round(item / scale, round_digits) for item in values)
        return f"metric-shape:{metric_name}:{_stable_digest(normalized)}"
    return f"metric-empty:{metric_name}"


def logical_observation_key(observation: CanonicalObservation) -> str:
    """Stable operational-fact key for evolving aggregate observations."""

    payload = _thaw(observation.payload)
    metadata = observation.metadata
    if observation.channel is TelemetryChannel.METRIC:
        resource = payload.get("resource")
        resource = resource if isinstance(resource, Mapping) else {}
        entity = resource.get("entity_id") or next(
            (item.entity_id for item in metadata.entities), "unknown"
        )
        return f"metric:{payload.get('metric_name', 'unknown')}:{entity}"
    if observation.channel is TelemetryChannel.ALERT:
        return (
            f"alert:{payload.get('alert_fingerprint', payload.get('alert_type', 'unknown'))}:"
            f"{payload.get('target', 'unknown')}"
        )
    if observation.channel is TelemetryChannel.LOG:
        entity_scope = ",".join(
            sorted({item.entity_id for item in metadata.entities})
        ) or str(payload.get("scope", "unknown"))
        return (
            f"log:{metadata.primary_subsystem}:"
            f"{payload.get('template_id', payload.get('event_type', 'unknown'))}:"
            f"{entity_scope}"
        )
    if observation.channel is TelemetryChannel.TRACE:
        return (
            f"trace:{payload.get('operation', 'unknown')}:"
            f"{payload.get('source', 'unknown')}:{payload.get('destination', 'unknown')}"
        )
    if observation.channel is TelemetryChannel.CONFIG:
        return (
            f"config:{payload.get('scope', 'unknown')}:"
            f"{payload.get('path', 'unknown')}"
        )
    return f"{observation.channel.value}:{observation.observation_id}"


def _request_is_targeted(request: AgentObservationRequest | None) -> bool:
    if request is None:
        return False
    return bool(
        request.metric_names
        or request.entity_ids
        or getattr(request, "subsystem_ids", ())
        or getattr(request, "alert_names", ())
        or request.detail == "raw"
        or request.requested_channels != _DEFAULT_REQUEST.requested_channels
        or request.lookback_seconds != _DEFAULT_REQUEST.lookback_seconds
    )


def request_is_targeted(request: AgentObservationRequest | None) -> bool:
    """Public predicate used by selection and diagnostics."""

    return _request_is_targeted(request)


def _request_match(
    observation: CanonicalObservation,
    *,
    metric_name: str | None,
    alert_name: str | None,
    entity_ids: Sequence[str],
    request: AgentObservationRequest | None,
    query_time_seconds: float,
    request_eligible: bool,
) -> bool:
    if not request_eligible or not _request_is_targeted(request):
        return False
    assert request is not None
    if observation.channel.value not in set(request.requested_channels):
        return False
    if (
        observation.channel is not TelemetryChannel.CONFIG
        and observation.metadata.event_end_time_seconds
        < max(0.0, query_time_seconds - request.lookback_seconds)
    ):
        return False
    if request.metric_names and (
        observation.channel is not TelemetryChannel.METRIC
        or metric_name not in set(request.metric_names)
    ):
        return False
    if request.entity_ids and not (set(request.entity_ids) & set(entity_ids)):
        return False
    subsystem_ids = set(getattr(request, "subsystem_ids", ()))
    if subsystem_ids and observation.metadata.primary_subsystem not in subsystem_ids:
        return False
    alert_names = set(getattr(request, "alert_names", ()))
    if alert_names and (
        observation.channel is not TelemetryChannel.ALERT
        or alert_name not in alert_names
    ):
        return False
    return True


def _is_slo_metric(
    metric_name: str,
    last: float | None,
    *,
    epsilon: float,
) -> bool:
    normalized = metric_name.lower().replace("-", "_")
    return (
        last is not None
        and last > epsilon
        and any(term in normalized for term in _SLO_TERMS)
    )


def _base_signals(
    observation: CanonicalObservation,
    *,
    query_time_seconds: float,
    request: AgentObservationRequest | None,
    config: StateBundleInferenceConfig,
    request_eligible: bool,
) -> ObservableSignals:
    payload = _thaw(observation.payload)
    visible_entities = {item.entity_id for item in observation.metadata.entities}
    resource = payload.get("resource")
    if isinstance(resource, Mapping):
        resource_entity = resource.get("entity_id")
        if isinstance(resource_entity, str) and resource_entity.strip():
            visible_entities.add(resource_entity.strip())
    for entity_field in ("target", "source", "destination", "scope"):
        entity_value = payload.get(entity_field)
        if isinstance(entity_value, str) and entity_value.strip():
            visible_entities.add(entity_value.strip())
    entity_ids = tuple(sorted(visible_entities))
    role_entities: dict[EntityRole, set[str]] = defaultdict(set)
    for item in observation.metadata.entities:
        role_entities[item.role].add(item.entity_id)
    role_targets = {
        item.entity_id
        for item in observation.metadata.entities
        if item.role is EntityRole.TARGET
        and item.entity_id.lower() not in _NON_SPECIFIC_TARGET_IDS
    }
    payload_target = payload.get("target")
    if (
        isinstance(payload_target, str)
        and payload_target.strip()
        and payload_target.strip().lower() not in _NON_SPECIFIC_TARGET_IDS
    ):
        role_targets.add(payload_target.strip())
    target_ids = tuple(sorted(role_targets))
    producer_ids = tuple(sorted(role_entities[EntityRole.PRODUCER]))
    affected_ids = tuple(sorted(role_entities[EntityRole.AFFECTED]))
    source_ids = tuple(sorted(role_entities[EntityRole.SOURCE]))
    destination_ids = tuple(sorted(role_entities[EntityRole.DESTINATION]))
    scope_ids = tuple(sorted(role_entities[EntityRole.SCOPE]))
    concrete_entities = tuple(
        item
        for item in entity_ids
        if item.lower() not in _GENERIC_ENTITY_IDS and item.lower() != "unknown"
    )
    metric_name: str | None = None
    alert_name: str | None = None
    severity: str | None = None
    active_alert = False
    slo_violation = False
    anomalous_metric = False
    zero_series = False
    constant_series = False
    recent_config_change = False
    static_config = False
    stale_config = False
    bookkeeping_log = False
    semantic_group: str

    if observation.channel is TelemetryChannel.METRIC:
        metric_name = str(payload.get("metric_name", "unknown"))
        values = _numeric_values(payload)
        zero_count = sum(abs(item) <= config.zero_absolute_epsilon for item in values)
        zero_series = bool(values) and (
            zero_count / len(values) >= config.zero_fraction_threshold
        )
        if values:
            mean = sum(values) / len(values)
            variance = sum((item - mean) ** 2 for item in values) / len(values)
            constant_series = variance <= config.constant_variance_epsilon
        last = _last_metric_value(payload, values)
        anomalous_metric = _metric_anomalous(payload, values, config)
        slo_violation = _is_slo_metric(
            metric_name,
            last,
            epsilon=config.zero_absolute_epsilon,
        )
        semantic_group = _metric_semantic_group(
            metric_name,
            values,
            zero_series=zero_series,
            round_digits=config.metric_shape_round_digits,
        )
    elif observation.channel is TelemetryChannel.ALERT:
        alert_name = str(payload.get("alert_type", "unknown"))
        severity = str(payload.get("severity", "unknown")).lower()
        active_alert = str(payload.get("status", "")).lower() == "firing"
        alert_text = _normalized_text(
            {
                "type": alert_name,
                "message": payload.get("message"),
                "details": payload.get("details"),
                "threshold": payload.get("threshold"),
            }
        ).replace(" ", "_")
        slo_violation = active_alert and any(term in alert_text for term in _SLO_TERMS)
        semantic_group = (
            f"alert:{alert_name}:{payload.get('status', 'unknown')}:"
            f"{payload.get('target', 'unknown')}"
        )
    elif observation.channel is TelemetryChannel.CONFIG:
        operation = str(payload.get("operation", "state")).lower()
        previous_value = payload.get("previous_value")
        change_time = payload.get("change_time_seconds")
        change_time = (
            float(change_time)
            if isinstance(change_time, (int, float))
            and not isinstance(change_time, bool)
            else observation.metadata.event_end_time_seconds
        )
        age = max(0.0, query_time_seconds - change_time)
        initial_baseline = previous_value is None and change_time <= 0.0
        value_changed = operation == "remove" or previous_value != payload.get("value")
        recent_config_change = (
            operation in {"set", "update", "remove"}
            and not initial_baseline
            and value_changed
            and age <= config.recent_config_window_seconds
        )
        static_config = (
            operation == "state"
            or initial_baseline
            or (operation in {"set", "update"} and not value_changed)
        )
        stale_config = static_config and age >= config.stale_config_age_seconds
        semantic_group = (
            f"config:{payload.get('scope', 'unknown')}:"
            f"{payload.get('path', 'unknown')}:"
            f"{_stable_digest(payload.get('value'))}"
        )
    elif observation.channel is TelemetryChannel.LOG:
        event_type = str(payload.get("event_type", "")).lower()
        template = str(payload.get("template", "")).lower()
        bookkeeping_log = any(
            term in event_type or term in template for term in _BOOKKEEPING_TERMS
        )
        semantic_group = (
            f"log:{observation.metadata.primary_subsystem}:"
            f"{payload.get('template_id', event_type)}"
        )
    else:
        status = payload.get("status_counts", {})
        semantic_group = (
            f"trace:{payload.get('operation', 'unknown')}:"
            f"{payload.get('source', 'unknown')}:"
            f"{payload.get('destination', 'unknown')}:"
            f"{_stable_digest(status)}"
        )

    request_match = _request_match(
        observation,
        metric_name=metric_name,
        alert_name=alert_name,
        entity_ids=entity_ids,
        request=request,
        query_time_seconds=query_time_seconds,
        request_eligible=request_eligible,
    )
    return ObservableSignals(
        logical_key=logical_observation_key(observation),
        semantic_group=semantic_group,
        entity_ids=entity_ids,
        target_ids=target_ids,
        producer_ids=producer_ids,
        affected_ids=affected_ids,
        source_ids=source_ids,
        destination_ids=destination_ids,
        scope_ids=scope_ids,
        metric_name=metric_name,
        alert_name=alert_name,
        severity=severity,
        active_alert=active_alert,
        target_bearing=bool(target_ids),
        concrete_entity=bool(concrete_entities),
        slo_violation=slo_violation,
        anomalous_metric=anomalous_metric,
        zero_series=zero_series,
        constant_series=constant_series,
        recent_config_change=recent_config_change,
        static_config=static_config,
        stale_config=stale_config,
        bookkeeping_log=bookkeeping_log,
        request_eligible=request_eligible,
        request_match=request_match,
    )


def analyze_observations(
    observations: Sequence[CanonicalObservation],
    *,
    query_time_seconds: float,
    request: AgentObservationRequest | None,
    config: StateBundleInferenceConfig,
) -> tuple[ObservableSignals, ...]:
    """Annotate a causal snapshot and derive only explicit cross-row links."""

    effective_request = request or _DEFAULT_REQUEST
    eligible_ids = {
        str(item.get("observation_id"))
        for item in filter_observations(
            [item.to_agent_dict(config.redaction_policy) for item in observations],
            query_time_seconds=query_time_seconds,
            request=effective_request,
        )
    }
    signals = [
        _base_signals(
            observation,
            query_time_seconds=query_time_seconds,
            request=request,
            config=config,
            request_eligible=observation.observation_id in eligible_ids,
        )
        for observation in observations
    ]
    active_alerts = [
        (observation, signal)
        for observation, signal in zip(observations, signals)
        if signal.active_alert
    ]
    active_entities = {
        entity
        for _, signal in active_alerts
        for entity in (*signal.entity_ids, *signal.target_ids)
    }
    anomalous_metric_names = {
        signal.metric_name
        for signal in signals
        if signal.metric_name is not None and signal.anomalous_metric
    }

    # Preserve each alert's fields, entity scope, subsystem, correlations, and
    # time interval.  A flattened global term union was the source of the
    # policy-v1 fan-out: a generic field from one alert could protect unrelated
    # rows associated with another alert.
    alert_link_records: list[dict[str, Any]] = []
    for alert, alert_signal in active_alerts:
        payload = _thaw(alert.payload)
        target = payload.get("target")
        alert_entities = set((*alert_signal.entity_ids, *alert_signal.target_ids))
        fields: list[dict[str, Any]] = []
        for source_name in ("details", "threshold"):
            for field_name, value in _flatten_alert_fields(payload.get(source_name)):
                terms = _normalized_measure_terms(field_name)
                fields.append(
                    {
                        "source": source_name,
                        "field": field_name,
                        "value": value,
                        "measure_terms": terms - _ENTITY_SCOPE_TERMS,
                        "scope_hints": terms & _ENTITY_SCOPE_TERMS,
                    }
                )
        alert_link_records.append(
            {
                "observation": alert,
                "signal": alert_signal,
                "target": str(target) if isinstance(target, str) else None,
                "entities": alert_entities,
                "concrete_entities": {
                    entity
                    for entity in alert_entities
                    if entity.lower() not in _GENERIC_ENTITY_IDS
                    and entity.lower() not in {"default", "unknown"}
                },
                "subsystem": alert.metadata.primary_subsystem.lower(),
                "correlations": set(alert.metadata.correlation_ids.items()),
                "fields": fields,
                "symptom_terms": _normalized_measure_terms(
                    {
                        "type": alert_signal.alert_name,
                        "message": payload.get("message"),
                    }
                ),
                "component_local": _component_local_alert(alert, alert_signal),
            }
        )

    def abnormal_operational_state(
        observation: CanonicalObservation, signal: ObservableSignals
    ) -> bool:
        payload = _thaw(observation.payload)
        if observation.channel is TelemetryChannel.METRIC:
            return signal.anomalous_metric or signal.slo_violation
        if observation.channel is TelemetryChannel.LOG:
            return str(payload.get("severity", "")).lower() in {
                "critical",
                "error",
                "high",
                "warning",
            }
        if observation.channel is TelemetryChannel.TRACE:
            status = payload.get("status_counts")
            return isinstance(status, Mapping) and any(
                str(key).lower() not in {"ok", "success", "successful"}
                and isinstance(value, (int, float))
                and not isinstance(value, bool)
                and float(value) > 0.0
                for key, value in status.items()
            )
        if observation.channel is TelemetryChannel.CONFIG:
            return signal.recent_config_change
        return signal.active_alert

    by_metric: dict[str, list[int]] = defaultdict(list)
    updated: list[ObservableSignals] = []
    for index, (observation, signal) in enumerate(zip(observations, signals)):
        best_match: tuple[Any, ...] | None = None
        metric_values: tuple[float, ...] = ()
        metric_last: float | None = None
        metric_measure: set[str] = set()
        metric_hints: set[str] = set()
        if observation.channel is TelemetryChannel.METRIC:
            payload = _thaw(observation.payload)
            metric_values = _numeric_values(payload)
            metric_last = _last_metric_value(payload, metric_values)
            metric_measure, metric_hints = _metric_measure_terms(
                signal.metric_name or ""
            )
            if signal.metric_name is not None:
                by_metric[signal.metric_name].append(index)

        observation_correlations = set(observation.metadata.correlation_ids.items())
        observation_entities = set(signal.entity_ids)
        observation_subsystem = observation.metadata.primary_subsystem.lower()
        for record in alert_link_records:
            alert = record["observation"]
            if alert.observation_id == observation.observation_id:
                continue
            if not _windows_compatible(
                observation,
                alert,
                max_gap_seconds=config.alert_link_max_gap_seconds,
            ):
                continue
            alert_entities = set(record["entities"])
            exact_entities = observation_entities & alert_entities
            exact_concrete = exact_entities & set(record["concrete_entities"])
            same_subsystem = observation_subsystem == record["subsystem"]
            correlation_match = bool(
                observation_correlations & set(record["correlations"])
            )
            state_compatible = abnormal_operational_state(observation, signal)

            match_strength = "no_match"
            match_reason: str | None = None
            matched_field: str | None = None
            matched_source_priority = 0
            matched_scope_specificity = 0
            semantic_overlap = 0
            if observation.channel is TelemetryChannel.METRIC and metric_measure:
                for field in record["fields"]:
                    field_measure = set(field["measure_terms"])
                    overlap = metric_measure & field_measure
                    required = min(
                        config.alert_metric_min_shared_terms,
                        len(metric_measure),
                        len(field_measure),
                    )
                    semantically_compatible = bool(required) and len(overlap) >= required
                    if not semantically_compatible:
                        continue
                    field_hints = set(field["scope_hints"])
                    contextual_scope_match = _scope_hint_matches_metric(
                        field_hints, metric_hints, signal.entity_ids
                    )
                    scoped_field = contextual_scope_match and bool(
                        field_hints
                        & {
                            "control",
                            "controlplane",
                            "scheduler",
                            "service",
                            "tenant",
                            "workload",
                        }
                    )
                    alert_target = str(record["target"] or "").lower()
                    default_application_scope = (
                        alert_target == "default"
                        and same_subsystem
                        and observation_subsystem == "application"
                        and "workload" in {item.lower() for item in signal.entity_ids}
                    )
                    aggregate_scope = (
                        alert_target in {"datacenter", "global"}
                        and bool(
                            {item.lower() for item in signal.entity_ids}
                            & {"datacenter", "global"}
                        )
                    )
                    entity_scope_compatible = bool(
                        exact_concrete
                        or scoped_field
                        or default_application_scope
                        or aggregate_scope
                    )
                    scope_specificity = (
                        4
                        if exact_concrete
                        else (
                            3
                            if default_application_scope
                            else (
                                2
                                if scoped_field
                                else (
                                    1
                                    if aggregate_scope or contextual_scope_match
                                    else 0
                                )
                            )
                        )
                    )
                    value_counterpart = _numeric_counterpart(
                        field["value"],
                        metric_last,
                        relative_tolerance=(
                            config.alert_counterpart_relative_tolerance
                        ),
                        epsilon=config.zero_absolute_epsilon,
                    )
                    nonzero_violation = not (
                        isinstance(field["value"], (int, float))
                        and not isinstance(field["value"], bool)
                        and abs(float(field["value"]))
                        <= config.zero_absolute_epsilon
                        and not signal.anomalous_metric
                    )
                    if (
                        value_counterpart
                        and nonzero_violation
                        and entity_scope_compatible
                        and (same_subsystem or scoped_field or exact_concrete)
                    ):
                        candidate_strength = "direct_counterpart"
                        candidate_reason = (
                            "alert_detail_value_counterpart"
                            if field["source"] == "details"
                            else "alert_threshold_value_counterpart"
                        )
                    elif (
                        state_compatible
                        and exact_concrete
                        and same_subsystem
                    ):
                        candidate_strength = "entity_and_symptom_match"
                        candidate_reason = "concrete_entity_subsystem_symptom_overlap"
                    else:
                        candidate_strength = "weak_context_match"
                        candidate_reason = "semantic_context_without_strong_scope"
                    candidate_key = (
                        _LINK_STRENGTH_PRIORITY[candidate_strength],
                        int(field["source"] == "details"),
                        scope_specificity,
                        len(overlap),
                        str(field["field"]),
                    )
                    current_key = (
                        _LINK_STRENGTH_PRIORITY[match_strength],
                        matched_source_priority,
                        matched_scope_specificity,
                        semantic_overlap,
                        matched_field or "",
                    )
                    if candidate_key > current_key:
                        match_strength = candidate_strength
                        match_reason = candidate_reason
                        matched_field = str(field["field"])
                        matched_source_priority = int(field["source"] == "details")
                        matched_scope_specificity = scope_specificity
                        semantic_overlap = len(overlap)

            if (
                correlation_match
                and state_compatible
                and (same_subsystem or bool(exact_entities))
            ):
                match_strength = "direct_counterpart"
                match_reason = "explicit_correlation_identifier"
                matched_field = matched_field or "correlation_id"
                matched_source_priority = 2
                matched_scope_specificity = 5
            elif match_strength == "no_match" and observation.channel is not TelemetryChannel.METRIC:
                observation_terms = _normalized_measure_terms(_thaw(observation.payload))
                symptom_overlap = observation_terms & set(record["symptom_terms"])
                if (
                    state_compatible
                    and exact_concrete
                    and same_subsystem
                    and len(symptom_overlap) >= config.alert_metric_min_shared_terms
                ):
                    match_strength = "entity_and_symptom_match"
                    match_reason = "concrete_entity_subsystem_symptom_overlap"
                    semantic_overlap = len(symptom_overlap)
                elif symptom_overlap and (same_subsystem or exact_entities):
                    match_strength = "weak_context_match"
                    match_reason = "semantic_context_without_strong_scope"
                    semantic_overlap = len(symptom_overlap)

            match_record = {
                "strength": match_strength,
                "reason": match_reason,
                "field": matched_field,
                "alert_entity": record["target"]
                or (sorted(alert_entities)[0] if alert_entities else None),
                "alert_observation_id": alert.observation_id,
                "alert_entities": alert_entities,
            }
            record_key = (
                _LINK_STRENGTH_PRIORITY[match_strength],
                matched_source_priority,
                matched_scope_specificity,
                semantic_overlap,
                alert.observation_id,
                match_record,
            )
            if best_match is None or record_key[:-1] > best_match[:-1]:
                best_match = record_key

        match = best_match[-1] if best_match is not None else {
            "strength": "no_match",
            "reason": None,
            "field": None,
            "alert_entity": None,
            "alert_observation_id": None,
            "alert_entities": set(),
        }
        linked = match["strength"] in {
            "direct_counterpart",
            "entity_and_symptom_match",
        }
        inferred_target_ids: tuple[str, ...] = ()
        if linked:
            inferred_target_ids = tuple(
                sorted(
                    entity
                    for entity in signal.entity_ids
                    if entity.lower() not in _NON_SPECIFIC_TARGET_IDS
                )
            )
        protection: list[str] = []
        if signal.request_match:
            protection.append("targeted_request")
        if signal.active_alert and signal.severity in {"critical", "high"}:
            protection.append("active_severe_alert")
        if signal.active_alert and signal.target_bearing:
            protection.append("active_target_alert")
        if signal.slo_violation:
            protection.append("violated_slo_or_health")
        if linked:
            protection.append("explicit_active_alert_link")
        updated.append(
            replace(
                signal,
                linked_to_active_alert=linked,
                alert_linkage_strength=str(match["strength"]),
                alert_linkage_reason=(
                    str(match["reason"]) if match["reason"] is not None else None
                ),
                matched_alert_field=(
                    str(match["field"]) if match["field"] is not None else None
                ),
                matched_alert_entity=(
                    str(match["alert_entity"])
                    if match["alert_entity"] is not None
                    else None
                ),
                matched_alert_observation_id=(
                    str(match["alert_observation_id"])
                    if match["alert_observation_id"] is not None
                    else None
                ),
                target_ids=tuple(
                    sorted(set((*signal.target_ids, *inferred_target_ids)))
                ),
                target_bearing=signal.target_bearing or bool(inferred_target_ids),
                protection_reasons=tuple(protection),
            )
        )

    # Estimate target roles from current observable relations.  These are
    # explicitly non-oracle selector estimates: component-local active alerts
    # establish candidate scopes; same-scope abnormal counterparts are direct,
    # while abnormal service/workload effects outside those scopes are marked
    # downstream.  In the absence of a competing component-local signal, an
    # abnormal row remains a direct candidate rather than being declared an
    # effect without evidence.
    direct_alert_indices = {
        index
        for index, (observation, signal) in enumerate(zip(observations, updated))
        if signal.active_alert and _component_local_alert(observation, signal)
    }
    direct_alert_entities = {
        entity
        for index in direct_alert_indices
        for entity in updated[index].entity_ids
    }
    direct_alert_subsystems = {
        observations[index].metadata.primary_subsystem.lower()
        for index in direct_alert_indices
    }
    role_updated: list[ObservableSignals] = []
    for index, (observation, signal) in enumerate(zip(observations, updated)):
        abnormal = abnormal_operational_state(observation, signal)
        same_direct_entity = bool(set(signal.entity_ids) & direct_alert_entities)
        same_direct_subsystem = (
            observation.metadata.primary_subsystem.lower() in direct_alert_subsystems
        )
        if index in direct_alert_indices or signal.alert_linkage_strength in {
            "direct_counterpart",
            "entity_and_symptom_match",
        }:
            role = "direct_target_candidate"
        elif signal.active_alert:
            role = (
                "downstream_affected_scope"
                if direct_alert_indices
                else "direct_target_candidate"
            )
        elif abnormal and (same_direct_entity or same_direct_subsystem):
            role = "direct_target_candidate"
        elif signal.affected_ids:
            role = "downstream_affected_scope"
        elif abnormal and direct_alert_indices:
            role = "downstream_affected_scope"
        elif abnormal or signal.recent_config_change:
            role = "direct_target_candidate"
        else:
            role = "contextual_peer"
        scopes = _scope_estimates(observation, signal, role=role)
        role_updated.append(
            replace(
                signal,
                estimated_target_role=role,
                estimated_scope_ids=scopes,
            )
        )
    updated = role_updated

    # Keep a bounded number of normal peers useful for entity contrast.  This
    # is a small ranking bonus, never blanket protection.
    for metric_name, indices in by_metric.items():
        diagnostic_entities = {
            entity
            for metric_index in indices
            if updated[metric_index].anomalous_metric
            or updated[metric_index].linked_to_active_alert
            or bool(set(updated[metric_index].entity_ids) & active_entities)
            for entity in updated[metric_index].entity_ids
        }
        if metric_name not in anomalous_metric_names and not diagnostic_entities:
            continue
        for metric_index in indices:
            signal = updated[metric_index]
            if (
                not signal.anomalous_metric
                and not signal.slo_violation
                and bool(set(signal.entity_ids) - diagnostic_entities)
            ):
                updated[metric_index] = replace(signal, normal_comparison=True)

    return tuple(
        replace(
            signal,
            equivalence_group_id=_equivalence_group_id(observation, signal),
        )
        for observation, signal in zip(observations, updated)
    )


def reranking_components(
    signal: ObservableSignals,
    config: StateBundleInferenceConfig,
    *,
    sticky: bool,
    post_action: bool,
    redundant: bool,
) -> tuple[tuple[str, float], ...]:
    """Return every non-learned score contribution for audit and ranking."""

    components: list[tuple[str, float]] = []

    def add(name: str, enabled: bool, value: float) -> None:
        if enabled and value:
            components.append((name, float(value)))

    add("active_alert", signal.active_alert, config.active_alert_bonus)
    add(
        "critical_severity",
        signal.active_alert and signal.severity == "critical",
        config.critical_alert_bonus,
    )
    add(
        "high_severity",
        signal.active_alert and signal.severity == "high",
        config.high_alert_bonus,
    )
    add(
        "warning_severity",
        signal.active_alert and signal.severity == "warning",
        config.warning_alert_bonus,
    )
    add("target_role", signal.target_bearing, config.target_role_bonus)
    add("concrete_entity", signal.concrete_entity, config.entity_identifier_bonus)
    add("slo_violation", signal.slo_violation, config.slo_violation_bonus)
    add("targeted_request", signal.request_match, config.targeted_request_bonus)
    add("sticky_evidence", sticky, config.sticky_evidence_bonus)
    add("post_action_evidence", post_action, config.post_action_evidence_bonus)
    add(
        "anomalous_metric",
        signal.anomalous_metric or signal.linked_to_active_alert,
        config.anomalous_metric_bonus,
    )
    add(
        "recent_config_change",
        signal.recent_config_change,
        config.recent_config_change_bonus,
    )
    add("normal_comparison", signal.normal_comparison, config.normal_comparison_bonus)
    add("zero_series", signal.zero_series, -config.zero_series_penalty)
    add(
        "constant_series",
        signal.constant_series and not signal.zero_series,
        -config.constant_series_penalty,
    )
    add("semantic_redundancy", redundant, -config.redundant_observation_penalty)
    add("static_config", signal.static_config, -config.static_config_penalty)
    add("stale_config", signal.stale_config, -config.stale_config_penalty)
    add("bookkeeping_log", signal.bookkeeping_log, -config.bookkeeping_log_penalty)
    return tuple(components)


__all__ = [
    "ObservableSignals",
    "analyze_observations",
    "logical_observation_key",
    "request_is_targeted",
    "reranking_components",
]
