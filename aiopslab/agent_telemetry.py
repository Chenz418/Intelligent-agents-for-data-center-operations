"""Deterministic agent-facing rendering for canonical telemetry.

The canonical ``statebundle.canonical.v1`` object is the authoritative
inference representation.  This module never mutates that object and
does not participate in collection, dataset generation, encoding, pooling, or
evidence selection.  It projects canonical observations into compact tables
only at the operations-agent boundary.

The projection deliberately has its own schema version.  In particular,
source references, correlations, repeated provenance, per-observation causal
cut tokens, and default quality values remain available in the canonical
object while being absent from the agent serialization.
"""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import importlib
import json
import math
import statistics
from typing import Any, Iterable, Mapping, Sequence


CANONICAL_SCHEMA_VERSION = "statebundle.canonical.v1"
STATEBUNDLE_OUTPUT_SCHEMA_VERSION = "statebundle.output.v1"
AGENT_SCHEMA_VERSION = "agent.telemetry.compact.v1"
AGENT_DELTA_SCHEMA_VERSION = "agent.telemetry.delta.v1"
CANONICAL_CHANNELS = ("log", "metric", "alert", "trace", "config")
CHANNEL_ORDER = {channel: index for index, channel in enumerate(CANONICAL_CHANNELS)}
DEFAULT_LOOKBACK_SECONDS = 300.0
DEFAULT_RECENT_SECONDS = 30.0
DEFAULT_LOG_LIMIT = 20
DEFAULT_LOG_VARIABLE_LIMIT = 12
_ESTIMATED_TARGET_ROLES = frozenset(
    {
        "direct_target_candidate",
        "downstream_affected_scope",
        "contextual_peer",
    }
)


_COMMON_COLUMNS = (
    "id",
    "key",
    "start",
    "end",
    "available",
    "subsystem",
    "entities",
    "quality",
)
_COLUMNS: dict[str, tuple[str, ...]] = {
    "log": _COMMON_COLUMNS
    + (
        "unit_type",
        "template_id",
        "template",
        "event_type",
        "count",
        "severity",
        "severity_histogram",
        "rarity",
        "burst_per_minute",
        "variables",
        "variable_count",
        "time_features",
    ),
    "metric": _COMMON_COLUMNS
    + (
        "unit_type",
        "name",
        "entity",
        "unit",
        "scale",
        "period_seconds",
        "recent",
        "baseline",
        "statistics",
        "normalization",
    ),
    "alert": (
        "group_key",
        "alert_type",
        "message",
        "status",
        "severity",
        "threshold",
        "details",
        "unit_type",
        "subsystem",
        "members",
    ),
    "trace": _COMMON_COLUMNS
    + (
        "unit_type",
        "operation",
        "source",
        "destination",
        "count",
        "status_counts",
        "retry_count",
        "latency_ms",
        "critical_path",
    ),
    "config": _COMMON_COLUMNS
    + (
        "unit_type",
        "scope",
        "path",
        "operation",
        "value_type",
        "previous_value",
        "value",
        "change_time",
    ),
}
_RAW_METRIC_COLUMNS = _COMMON_COLUMNS + (
    "unit_type",
    "name",
    "entity",
    "unit",
    "scale",
    "period_seconds",
    "timestamps",
    "values",
    "missingness_mask",
    "statistics",
    "normalization",
)
_ALERT_MEMBER_COLUMNS = (
    "id",
    "key",
    "start",
    "end",
    "available",
    "fingerprint",
    "target",
    "duration",
    "entities",
    "quality",
)
_METRIC_RECENT_COLUMNS = ("axis", "values", "missing_indices")
_METRIC_BASELINE_COLUMNS = (
    "start",
    "end",
    "sample_count",
    "observed_count",
    "median",
    "q1",
    "q3",
    "last",
    "trend_per_second",
    "change_points",
)
_METRIC_STATISTIC_COLUMNS = (
    "count",
    "last",
    "max",
    "mean",
    "median",
    "min",
    "p95",
    "slope_per_second",
    "stddev",
)
_METRIC_NORMALIZATION_COLUMNS = (
    "method",
    "sample_count",
    "median",
    "iqr",
    "last",
    "reference_end_time_seconds",
)


@dataclass(frozen=True, slots=True)
class AgentObservationRequest:
    """Validated controls for one agent-facing observation view."""

    include_config: bool = True
    channels: tuple[str, ...] | None = None
    lookback_seconds: float = DEFAULT_LOOKBACK_SECONDS
    log_limit: int | None = DEFAULT_LOG_LIMIT
    detail: str = "overview"
    metric_names: tuple[str, ...] = ()
    entity_ids: tuple[str, ...] = ()
    subsystem_ids: tuple[str, ...] = ()
    alert_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.include_config, bool):
            raise TypeError("include_config must be a boolean")
        channels = self.channels
        if channels is not None:
            channels = tuple(str(channel).strip() for channel in channels)
            unsupported = sorted(set(channels) - set(CANONICAL_CHANNELS))
            if unsupported:
                raise ValueError(f"unsupported telemetry channels: {unsupported}")
            if len(set(channels)) != len(channels):
                raise ValueError("channels must not contain duplicates")
            object.__setattr__(
                self,
                "channels",
                tuple(channel for channel in CANONICAL_CHANNELS if channel in channels),
            )
        if isinstance(self.lookback_seconds, bool) or not isinstance(
            self.lookback_seconds, (int, float)
        ):
            raise TypeError("lookback_seconds must be numeric")
        lookback = float(self.lookback_seconds)
        if not math.isfinite(lookback) or lookback < 0:
            raise ValueError("lookback_seconds must be a finite non-negative number")
        object.__setattr__(self, "lookback_seconds", lookback)
        if self.log_limit is not None and (
            isinstance(self.log_limit, bool)
            or not isinstance(self.log_limit, int)
            or self.log_limit < 0
        ):
            raise ValueError("log_limit must be a non-negative integer or null")
        if self.detail not in {"overview", "raw"}:
            raise ValueError("detail must be 'overview' or 'raw'")
        object.__setattr__(self, "metric_names", _stable_text_tuple(self.metric_names))
        object.__setattr__(self, "entity_ids", _stable_text_tuple(self.entity_ids))
        object.__setattr__(
            self, "subsystem_ids", _stable_text_tuple(self.subsystem_ids)
        )
        object.__setattr__(self, "alert_names", _stable_text_tuple(self.alert_names))

    @property
    def requested_channels(self) -> tuple[str, ...]:
        # ``()`` is an intentional request for no channels; only ``None`` means
        # the default complete channel set.
        channels = CANONICAL_CHANNELS if self.channels is None else self.channels
        if not self.include_config:
            channels = tuple(channel for channel in channels if channel != "config")
        return channels

    @property
    def view_key(self) -> tuple[Any, ...]:
        return (
            self.include_config,
            self.requested_channels,
            self.lookback_seconds,
            self.log_limit,
            self.detail,
            self.metric_names,
            self.entity_ids,
            self.subsystem_ids,
            self.alert_names,
        )


def _stable_text_tuple(value: Iterable[str] | None) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        value = (value,)
    result = tuple(sorted({str(item).strip() for item in value if str(item).strip()}))
    return result


def deterministic_json(value: Any) -> str:
    """Serialize one prompt payload with stable, whitespace-free JSON."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
        default=str,
    )


def estimate_tokens(value: Any, *, characters_per_token: float = 4.0) -> int:
    """Return the benchmark's deterministic tokenizer-independent estimate."""

    if characters_per_token <= 0:
        raise ValueError("characters_per_token must be positive")
    return max(1, math.ceil(len(deterministic_json(value)) / characters_per_token))


@lru_cache(maxsize=4)
def _token_encoding(encoding_name: str) -> Any:
    """Load the benchmark tokenizer lazily.

    ``tiktoken`` is a benchmark dependency, but keeping the import lazy avoids
    imposing tokenizer startup cost on canonical collection paths that never
    render agent telemetry.
    """

    tiktoken = importlib.import_module("tiktoken")
    return tiktoken.get_encoding(encoding_name)


def count_serialized_tokens(
    value: Any,
    *,
    encoding_name: str = "o200k_base",
) -> int:
    """Count the exact deterministic compact JSON tokens used by the benchmark."""

    encoded = deterministic_json(value)
    return len(_token_encoding(encoding_name).encode(encoded, disallowed_special=()))


def logical_observation_key(observation: Mapping[str, Any]) -> str:
    """Return a renderer-only stable series/entity key.

    Canonical observation IDs remain untouched.  This key exists solely to
    match evolving series/aggregates across snapshots for delta rendering.
    """

    channel = str(observation.get("channel", "unknown"))
    payload = observation.get("payload")
    payload = payload if isinstance(payload, Mapping) else {}
    metadata = observation.get("metadata")
    metadata = metadata if isinstance(metadata, Mapping) else {}
    subsystem = str(metadata.get("primary_subsystem", "unknown"))
    if channel == "metric":
        resource = payload.get("resource")
        resource = resource if isinstance(resource, Mapping) else {}
        entity = resource.get("entity_id") or _first_entity_id(metadata)
        parts = (payload.get("metric_name"), entity, resource.get("entity_type"))
    elif channel == "log":
        parts = (payload.get("template_id"), subsystem, _first_entity_id(metadata))
    elif channel == "alert":
        parts = (
            payload.get("alert_fingerprint") or payload.get("alert_type"),
            payload.get("target"),
        )
    elif channel == "trace":
        parts = (
            payload.get("operation"),
            payload.get("source"),
            payload.get("destination"),
        )
    elif channel == "config":
        parts = (payload.get("scope"), payload.get("path"))
    else:
        parts = (observation.get("observation_id"),)
    normalized = "|".join("" if part is None else str(part) for part in parts)
    return f"{channel}:{normalized}"


def _first_entity_id(metadata: Mapping[str, Any]) -> str | None:
    entities = metadata.get("entities")
    if not isinstance(entities, Sequence) or isinstance(entities, (str, bytes)):
        return None
    for entity in entities:
        if isinstance(entity, Mapping) and isinstance(entity.get("entity_id"), str):
            return str(entity["entity_id"])
    return None


def _compact_entities(
    metadata: Mapping[str, Any],
    *,
    exclude_entity_id: str | None = None,
) -> list[list[Any]] | None:
    entities = metadata.get("entities")
    if not isinstance(entities, Sequence) or isinstance(entities, (str, bytes)):
        return None
    compact: list[list[Any]] = []
    for entity in entities:
        if not isinstance(entity, Mapping):
            continue
        entity_id = entity.get("entity_id")
        if not isinstance(entity_id, str) or entity_id == exclude_entity_id:
            continue
        row: list[Any] = [entity_id, entity.get("role")]
        confidence = entity.get("confidence", 1.0)
        if confidence != 1 and confidence != 1.0:
            row.append(confidence)
        compact.append(row)
    return compact or None


def _compact_quality(metadata: Mapping[str, Any]) -> dict[str, Any] | None:
    quality = metadata.get("data_quality")
    if not isinstance(quality, Mapping):
        return None
    result: dict[str, Any] = {}
    parse_confidence = quality.get("parse_confidence", 1.0)
    missingness = quality.get("missingness_fraction", 0.0)
    delay = quality.get("delay_seconds", 0.0)
    if parse_confidence != 1 and parse_confidence != 1.0:
        result["parse_confidence"] = parse_confidence
    if missingness != 0 and missingness != 0.0:
        result["missingness_fraction"] = missingness
    if delay != 0 and delay != 0.0:
        result["delay_seconds"] = delay
    mask = quality.get("availability_mask")
    if isinstance(mask, Mapping):
        unavailable = sorted(
            str(key) for key, available in mask.items() if available is False
        )
        if unavailable:
            result["unavailable"] = unavailable
    flags = quality.get("validation_flags")
    if isinstance(flags, Sequence) and not isinstance(flags, (str, bytes)) and flags:
        result["validation_flags"] = list(flags)
    return result or None


def _common_values(observation: Mapping[str, Any]) -> list[Any]:
    metadata = observation.get("metadata")
    metadata = metadata if isinstance(metadata, Mapping) else {}
    window = observation.get("window")
    window = window if isinstance(window, Mapping) else {}
    return [
        observation.get("observation_id"),
        logical_observation_key(observation),
        window.get("start_time_seconds", metadata.get("event_start_time_seconds")),
        window.get("end_time_seconds", metadata.get("event_end_time_seconds")),
        metadata.get("available_at_time_seconds"),
        metadata.get("primary_subsystem"),
        _compact_entities(metadata),
        _compact_quality(metadata),
    ]


def _quantile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _trend(times: Sequence[float], values: Sequence[float | None]) -> float | None:
    points = [
        (float(time), float(value))
        for time, value in zip(times, values, strict=False)
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]
    if len(points) < 2:
        return 0.0 if points else None
    mean_time = statistics.fmean(time for time, _ in points)
    mean_value = statistics.fmean(value for _, value in points)
    denominator = sum((time - mean_time) ** 2 for time, _ in points)
    if denominator == 0:
        return 0.0
    return (
        sum((time - mean_time) * (value - mean_value) for time, value in points)
        / denominator
    )


def _change_points(
    times: Sequence[float], values: Sequence[float | None]
) -> list[list[float]]:
    """Return deterministic distribution-free level-change points.

    The rule is representational, not a relevance/anomaly filter: a point is
    retained when its adjacent change exceeds the robust spread (or any
    non-zero change for a flat baseline).  First/last values remain represented
    separately in the baseline tuple.
    """

    numeric = [
        float(value)
        for value in values
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]
    if len(numeric) < 2:
        return []
    q1 = _quantile(numeric, 0.25)
    q3 = _quantile(numeric, 0.75)
    spread = 0.0 if q1 is None or q3 is None else max(0.0, q3 - q1)
    threshold = spread
    result: list[list[float]] = []
    prior: float | None = None
    for time, value in zip(times, values, strict=False):
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            continue
        current = float(value)
        if prior is not None and abs(current - prior) > threshold:
            result.append([float(time), current])
        prior = current
    return result


def _metric_summary(
    times: Sequence[float], values: Sequence[float | None]
) -> list[Any] | None:
    if not times:
        return None
    numeric = [
        float(value)
        for value in values
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]
    return [
        float(times[0]),
        float(times[-1]),
        len(times),
        len(numeric),
        _quantile(numeric, 0.5),
        _quantile(numeric, 0.25),
        _quantile(numeric, 0.75),
        numeric[-1] if numeric else None,
        _trend(times, values),
        _change_points(times, values),
    ]


def _metric_statistics(
    times: Sequence[float], values: Sequence[float | None]
) -> list[Any] | None:
    """Recompute canonical-compatible statistics for a projected time slice."""

    observed = [
        (float(time), float(value))
        for time, value in zip(times, values, strict=False)
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]
    if not observed:
        return None
    numeric = [value for _, value in observed]
    slope = 0.0
    if len(observed) > 1 and observed[-1][0] != observed[0][0]:
        slope = (observed[-1][1] - observed[0][1]) / (observed[-1][0] - observed[0][0])
    return [
        float(len(numeric)),
        numeric[-1],
        max(numeric),
        statistics.fmean(numeric),
        _quantile(numeric, 0.5),
        min(numeric),
        _quantile(numeric, 0.95),
        slope,
        statistics.pstdev(numeric) if len(numeric) > 1 else 0.0,
    ]


def _metric_normalization_for_slice(
    *,
    before: Sequence[tuple[float, Any, bool]],
    current: Sequence[tuple[float, Any, bool]],
) -> list[Any] | None:
    reference = [
        (time, float(value))
        for time, value, missing in before
        if not missing
        and isinstance(value, (int, float))
        and not isinstance(value, bool)
    ]
    method = "causal_pre_window_robust"
    if not reference:
        reference = [
            (time, float(value))
            for time, value, missing in current
            if not missing
            and isinstance(value, (int, float))
            and not isinstance(value, bool)
        ]
        method = "causal_window_fallback"
    if not reference:
        return None
    values = [value for _, value in reference]
    q1 = _quantile(values, 0.25)
    q3 = _quantile(values, 0.75)
    return [
        method,
        len(values),
        _quantile(values, 0.5),
        None if q1 is None or q3 is None else q3 - q1,
        values[-1],
        reference[-1][0] if method == "causal_pre_window_robust" else None,
    ]


def _ordered_mapping_values(value: Any, columns: Sequence[str]) -> list[Any] | None:
    if not isinstance(value, Mapping):
        return None
    return [value.get(column) for column in columns]


def _metric_row(
    observation: Mapping[str, Any],
    *,
    query_time_seconds: float,
    recent_seconds: float,
    raw: bool,
    request_cutoff: float,
    previous: Mapping[str, Any] | None = None,
) -> list[Any]:
    payload = observation.get("payload")
    payload = payload if isinstance(payload, Mapping) else {}
    metadata = observation.get("metadata")
    metadata = metadata if isinstance(metadata, Mapping) else {}
    resource = payload.get("resource")
    resource = resource if isinstance(resource, Mapping) else {}
    entity_id = resource.get("entity_id") or _first_entity_id(metadata)
    entity = [entity_id, resource.get("entity_type")]
    if entity[-1] is None:
        entity.pop()
    timestamps = payload.get("timestamps_seconds")
    values = payload.get("values")
    mask = payload.get("missingness_mask")
    timestamps = (
        list(timestamps)
        if isinstance(timestamps, Sequence) and not isinstance(timestamps, (str, bytes))
        else []
    )
    values = (
        list(values)
        if isinstance(values, Sequence) and not isinstance(values, (str, bytes))
        else []
    )
    mask = (
        list(mask)
        if isinstance(mask, Sequence) and not isinstance(mask, (str, bytes))
        else []
    )
    all_triples = [
        (float(time), value, bool(mask[index]) if index < len(mask) else value is None)
        for index, (time, value) in enumerate(zip(timestamps, values, strict=False))
        if isinstance(time, (int, float)) and not isinstance(time, bool)
    ]
    excluded_triples = [item for item in all_triples if item[0] < request_cutoff]
    triples = [item for item in all_triples if item[0] >= request_cutoff]
    timestamps = [time for time, _, _ in triples]
    values = [value for _, value, _ in triples]
    mask = [missing for _, _, missing in triples]
    common = _common_values(observation)
    if excluded_triples and isinstance(common[2], (int, float)):
        common[2] = max(float(common[2]), request_cutoff)
    # The resource entity is already represented in the metric entity column.
    common[6] = _compact_entities(
        metadata, exclude_entity_id=str(entity_id) if entity_id else None
    )
    prefix = common + [
        payload.get("unit_type"),
        payload.get("metric_name"),
        entity,
        payload.get("unit"),
        payload.get("scale"),
        payload.get("sample_period_seconds"),
    ]
    if excluded_triples:
        statistics_values = _metric_statistics(timestamps, values)
        normalization_values = _metric_normalization_for_slice(
            before=excluded_triples,
            current=triples,
        )
    else:
        statistics_values = _ordered_mapping_values(
            payload.get("statistics"), _METRIC_STATISTIC_COLUMNS
        )
        normalization_values = _ordered_mapping_values(
            payload.get("normalization_reference"), _METRIC_NORMALIZATION_COLUMNS
        )
    if raw:
        return prefix + [
            timestamps,
            values,
            mask,
            statistics_values,
            normalization_values,
        ]

    recent_cutoff = max(request_cutoff, query_time_seconds - recent_seconds)
    changed_or_new_times: set[float] = set()
    if previous is not None:
        previous_payload = previous.get("payload")
        previous_payload = (
            previous_payload if isinstance(previous_payload, Mapping) else {}
        )
        previous_times = previous_payload.get("timestamps_seconds")
        previous_values = previous_payload.get("values")
        previous_mask = previous_payload.get("missingness_mask")
        if (
            isinstance(previous_times, Sequence)
            and not isinstance(previous_times, (str, bytes))
            and isinstance(previous_values, Sequence)
            and not isinstance(previous_values, (str, bytes))
        ):
            previous_mask = (
                list(previous_mask)
                if isinstance(previous_mask, Sequence)
                and not isinstance(previous_mask, (str, bytes))
                else []
            )
            previous_by_time = {
                float(time): (
                    value,
                    bool(previous_mask[index])
                    if index < len(previous_mask)
                    else value is None,
                )
                for index, (time, value) in enumerate(
                    zip(previous_times, previous_values, strict=False)
                )
                if isinstance(time, (int, float)) and not isinstance(time, bool)
            }
            changed_or_new_times = {
                time
                for time, value, missing in triples
                if time not in previous_by_time
                or previous_by_time[time] != (value, missing)
            }
    recent = [
        [time, value, missing]
        for time, value, missing in zip(timestamps, values, mask, strict=False)
        if (
            time in changed_or_new_times
            if previous is not None
            else time >= recent_cutoff
        )
    ]
    baseline_times = [
        time for time in timestamps if time < query_time_seconds - recent_seconds
    ]
    baseline_values = [
        value
        for time, value in zip(timestamps, values, strict=False)
        if time < query_time_seconds - recent_seconds
    ]
    baseline = _metric_summary(baseline_times, baseline_values)
    return prefix + [recent, baseline, statistics_values, normalization_values]


def _overview_value(value: Any, *, depth: int = 0) -> Any:
    """Bound nested log state while advertising exactly what was elided.

    Canonical log aggregates occasionally place a whole workload/system mapping
    below one variable key.  Top-level key capping alone would therefore leave
    a pathological multi-kilobyte cell.  Raw drill-down bypasses this preview.
    """

    if isinstance(value, Mapping):
        ordered = sorted(value.items(), key=lambda item: str(item[0]))
        if depth >= 2:
            return {"value_type": "object", "item_count": len(ordered)}
        retained = ordered[:8]
        result = {
            str(key): _overview_value(item, depth=depth + 1) for key, item in retained
        }
        if len(ordered) > len(retained):
            result["omitted_item_count"] = len(ordered) - len(retained)
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        retained = list(value[:8])
        result = [_overview_value(item, depth=depth + 1) for item in retained]
        if len(value) > len(retained):
            result.append({"omitted_item_count": len(value) - len(retained)})
        return result
    if isinstance(value, str) and len(value) > 256:
        return {"preview": value[:256], "character_count": len(value)}
    return value


def _bounded_variables(value: Any, limit: int | None) -> tuple[Any, int | None]:
    if not isinstance(value, Mapping):
        return value, None
    count = len(value)
    if limit is None or count <= limit:
        return (
            dict(sorted(value.items()))
            if limit is None
            else {key: _overview_value(item) for key, item in sorted(value.items())},
            count,
        )
    return {key: _overview_value(value[key]) for key in sorted(value)[:limit]}, count


def _non_alert_row(
    observation: Mapping[str, Any],
    *,
    query_time_seconds: float,
    recent_seconds: float,
    request_cutoff: float,
    raw: bool,
    log_variable_limit: int,
    previous: Mapping[str, Any] | None = None,
) -> list[Any]:
    channel = observation.get("channel")
    payload = observation.get("payload")
    payload = payload if isinstance(payload, Mapping) else {}
    if channel == "metric":
        return _metric_row(
            observation,
            query_time_seconds=query_time_seconds,
            recent_seconds=recent_seconds,
            raw=raw,
            request_cutoff=request_cutoff,
            previous=previous,
        )
    common = _common_values(observation)
    if channel == "log":
        variables, variable_count = _bounded_variables(
            payload.get("variable_summaries"),
            None if raw else log_variable_limit,
        )
        return common + [
            payload.get("unit_type"),
            payload.get("template_id"),
            payload.get("template"),
            payload.get("event_type"),
            payload.get("count"),
            payload.get("severity"),
            payload.get("severity_histogram"),
            payload.get("rarity"),
            payload.get("burst_rate_per_minute"),
            variables,
            variable_count,
            payload.get("time_features"),
        ]
    if channel == "trace":
        return common + [
            payload.get("unit_type"),
            payload.get("operation"),
            payload.get("source"),
            payload.get("destination"),
            payload.get("count"),
            payload.get("status_counts"),
            payload.get("retry_count"),
            payload.get("latency_ms"),
            payload.get("critical_path"),
        ]
    if channel == "config":
        return common + [
            payload.get("unit_type"),
            payload.get("scope"),
            payload.get("path"),
            payload.get("operation"),
            payload.get("value_type"),
            payload.get("previous_value"),
            payload.get("value"),
            payload.get("change_time_seconds"),
        ]
    raise ValueError(f"unsupported observation channel: {channel!r}")


def _alert_signature(observation: Mapping[str, Any]) -> tuple[str, ...]:
    payload = observation.get("payload")
    payload = payload if isinstance(payload, Mapping) else {}
    metadata = observation.get("metadata")
    metadata = metadata if isinstance(metadata, Mapping) else {}
    shared = {
        "alert_type": payload.get("alert_type"),
        "message": _alert_message_template(
            payload.get("message"), payload.get("target")
        ),
        "status": payload.get("status"),
        "severity": payload.get("severity"),
        "threshold": payload.get("threshold"),
        "details": _alert_shared_details(payload.get("details"), payload.get("target")),
        "unit_type": payload.get("unit_type"),
        "subsystem": metadata.get("primary_subsystem"),
    }
    return (deterministic_json(shared),)


def _alert_message_template(message: Any, target: Any) -> Any:
    if isinstance(message, str) and isinstance(target, str) and target:
        return message.replace(target, "{target}")
    return message


def _alert_shared_details(details: Any, target: Any) -> Any:
    if not isinstance(details, Mapping):
        return details
    return {
        key: ("{target}" if isinstance(target, str) and value == target else value)
        for key, value in sorted(details.items())
    }


def _alert_rows(observations: Sequence[Mapping[str, Any]]) -> list[list[Any]]:
    groups: dict[tuple[str, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for observation in observations:
        groups[_alert_signature(observation)].append(observation)
    rows: list[list[Any]] = []
    for signature in sorted(groups):
        members = sorted(
            groups[signature],
            key=lambda item: (
                _event_end(item),
                str(item.get("observation_id", "")),
            ),
        )
        first = members[0]
        payload = first.get("payload")
        payload = payload if isinstance(payload, Mapping) else {}
        metadata = first.get("metadata")
        metadata = metadata if isinstance(metadata, Mapping) else {}
        group_key = (
            "alert-group:"
            + hashlib.sha256(signature[0].encode("utf-8")).hexdigest()[:16]
        )
        member_rows: list[list[Any]] = []
        for observation in members:
            item_payload = observation.get("payload")
            item_payload = item_payload if isinstance(item_payload, Mapping) else {}
            item_metadata = observation.get("metadata")
            item_metadata = item_metadata if isinstance(item_metadata, Mapping) else {}
            window = observation.get("window")
            window = window if isinstance(window, Mapping) else {}
            member_rows.append(
                [
                    observation.get("observation_id"),
                    logical_observation_key(observation),
                    window.get("start_time_seconds"),
                    window.get("end_time_seconds"),
                    item_metadata.get("available_at_time_seconds"),
                    item_payload.get("alert_fingerprint"),
                    item_payload.get("target"),
                    item_payload.get("duration_seconds"),
                    _compact_entities(item_metadata),
                    _compact_quality(item_metadata),
                ]
            )
        rows.append(
            [
                group_key,
                payload.get("alert_type"),
                _alert_message_template(payload.get("message"), payload.get("target")),
                payload.get("status"),
                payload.get("severity"),
                payload.get("threshold"),
                _alert_shared_details(payload.get("details"), payload.get("target")),
                payload.get("unit_type"),
                metadata.get("primary_subsystem"),
                member_rows,
            ]
        )
    return rows


def _event_end(observation: Mapping[str, Any]) -> float:
    metadata = observation.get("metadata")
    metadata = metadata if isinstance(metadata, Mapping) else {}
    value = metadata.get("event_end_time_seconds")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    window = observation.get("window")
    window = window if isinstance(window, Mapping) else {}
    value = window.get("end_time_seconds", 0.0)
    return (
        float(value)
        if isinstance(value, (int, float)) and not isinstance(value, bool)
        else 0.0
    )


def filter_observations(
    observations: Iterable[Mapping[str, Any]],
    *,
    query_time_seconds: float,
    request: AgentObservationRequest,
) -> list[Mapping[str, Any]]:
    """Apply only explicit request controls; never relevance/anomaly filtering."""

    cutoff = max(0.0, query_time_seconds - request.lookback_seconds)
    allowed_channels = set(request.requested_channels)
    metric_names = set(request.metric_names)
    entity_ids = set(request.entity_ids)
    subsystem_ids = set(request.subsystem_ids)
    alert_names = set(request.alert_names)
    filtered: list[Mapping[str, Any]] = []
    for observation in observations:
        channel = observation.get("channel")
        # Configuration observations are canonical scoped state, not expiring
        # events.  A narrow canonical query intentionally includes the current
        # state even when its original change predates the lookback window.
        if channel not in allowed_channels or (
            channel != "config" and _event_end(observation) < cutoff
        ):
            continue
        payload = observation.get("payload")
        payload = payload if isinstance(payload, Mapping) else {}
        metadata = observation.get("metadata")
        metadata = metadata if isinstance(metadata, Mapping) else {}
        if subsystem_ids and metadata.get("primary_subsystem") not in subsystem_ids:
            continue
        if metric_names and (
            channel != "metric" or payload.get("metric_name") not in metric_names
        ):
            continue
        if alert_names and (
            channel != "alert" or payload.get("alert_type") not in alert_names
        ):
            continue
        if entity_ids:
            visible_entity_ids: set[str] = set()
            if channel == "metric":
                resource = payload.get("resource")
                resource = resource if isinstance(resource, Mapping) else {}
                entity_id = resource.get("entity_id") or _first_entity_id(metadata)
                if isinstance(entity_id, str):
                    visible_entity_ids.add(entity_id)
            for field in ("target", "source", "destination", "scope"):
                value = payload.get(field)
                if isinstance(value, str):
                    visible_entity_ids.add(value)
            entities = metadata.get("entities")
            if isinstance(entities, Sequence) and not isinstance(
                entities, (str, bytes)
            ):
                visible_entity_ids.update(
                    str(entity.get("entity_id"))
                    for entity in entities
                    if isinstance(entity, Mapping)
                    and isinstance(entity.get("entity_id"), str)
                )
            if not (entity_ids & visible_entity_ids):
                continue
        filtered.append(observation)
    logs = sorted(
        (item for item in filtered if item.get("channel") == "log"),
        key=lambda item: (-_event_end(item), str(item.get("observation_id", ""))),
    )
    if request.log_limit is not None:
        logs = logs[: request.log_limit]
    allowed_log_ids = {id(item) for item in logs}
    filtered = [
        item
        for item in filtered
        if item.get("channel") != "log" or id(item) in allowed_log_ids
    ]
    return sorted(
        filtered,
        key=lambda item: (
            CHANNEL_ORDER.get(str(item.get("channel")), len(CHANNEL_ORDER)),
            _event_end(item),
            str(item.get("observation_id", "")),
        ),
    )


def _table_metadata(
    raw_metrics: bool,
    present_channels: Iterable[str],
) -> dict[str, Any]:
    channels = set(present_channels)
    result: dict[str, Any] = {}
    if "metric" in channels:
        result.update(
            {
                "metric_recent_columns": list(_METRIC_RECENT_COLUMNS),
                "metric_baseline_columns": list(_METRIC_BASELINE_COLUMNS),
                "metric_statistic_columns": list(_METRIC_STATISTIC_COLUMNS),
                "metric_normalization_columns": list(_METRIC_NORMALIZATION_COLUMNS),
                "metric_mode": (
                    "raw_series" if raw_metrics else "recent_plus_baseline"
                ),
            }
        )
    if "alert" in channels:
        result["alert_member_columns"] = list(_ALERT_MEMBER_COLUMNS)
    return result


def render_tables(
    observations: Sequence[Mapping[str, Any]],
    *,
    query_time_seconds: float,
    request: AgentObservationRequest,
    recent_seconds: float = DEFAULT_RECENT_SECONDS,
    log_variable_limit: int = DEFAULT_LOG_VARIABLE_LIMIT,
    previous_by_key: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Render all supplied observations into shared-column channel tables."""

    request_cutoff = max(0.0, query_time_seconds - request.lookback_seconds)
    by_channel: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for observation in observations:
        by_channel[str(observation.get("channel"))].append(observation)
    tables: dict[str, dict[str, Any]] = {}
    for channel in CANONICAL_CHANNELS:
        items = by_channel.get(channel, [])
        if not items:
            continue
        if channel == "alert":
            rows = _alert_rows(items)
            columns = _COLUMNS[channel]
        else:
            columns = (
                _RAW_METRIC_COLUMNS
                if channel == "metric" and request.detail == "raw"
                else _COLUMNS[channel]
            )
            rows = [
                _non_alert_row(
                    item,
                    query_time_seconds=query_time_seconds,
                    recent_seconds=recent_seconds,
                    request_cutoff=request_cutoff,
                    raw=request.detail == "raw",
                    log_variable_limit=log_variable_limit,
                    previous=(previous_by_key or {}).get(logical_observation_key(item)),
                )
                for item in items
            ]
        table: dict[str, Any] = {"columns": list(columns), "rows": rows}
        if channel == "metric" and request.detail != "raw":
            recent_index = columns.index("recent")
            axes: list[list[float]] = []
            axis_index: dict[tuple[float, ...], int] = {}
            for row in rows:
                recent = row[recent_index]
                if not recent:
                    row[recent_index] = None
                    continue
                timestamps = tuple(float(point[0]) for point in recent)
                index = axis_index.get(timestamps)
                if index is None:
                    index = len(axes)
                    axis_index[timestamps] = index
                    axes.append(list(timestamps))
                values = [point[1] for point in recent]
                missing = [offset for offset, point in enumerate(recent) if point[2]]
                row[recent_index] = [index, values, missing or None]
            table["recent_time_axes"] = axes
        tables[channel] = table
    return tables


def compact_observation_fragment(
    observation: Mapping[str, Any],
    *,
    query_time_seconds: float | None = None,
    recent_seconds: float = DEFAULT_RECENT_SECONDS,
    request: AgentObservationRequest | None = None,
) -> dict[str, Any]:
    """Return the exact singleton table fragment used for budget estimates."""

    query_time = (
        _event_end(observation)
        if query_time_seconds is None
        else float(query_time_seconds)
    )
    source_request = request or AgentObservationRequest()
    fragment_request = AgentObservationRequest(
        channels=(str(observation.get("channel")),),
        include_config=source_request.include_config,
        lookback_seconds=source_request.lookback_seconds,
        log_limit=1,
        detail=source_request.detail,
    )
    tables = render_tables(
        [observation],
        query_time_seconds=query_time,
        request=fragment_request,
        recent_seconds=recent_seconds,
    )
    return tables[str(observation.get("channel"))]


def compact_statebundle_observation_token_cost(
    observation: Mapping[str, Any],
    *,
    query_time_seconds: float | None = None,
    encoding_name: str = "o200k_base",
    request: AgentObservationRequest | None = None,
) -> int:
    """Conservatively cost one row plus its StateBundle group association.

    The merged output shares table headers and one group envelope among several
    observations. Charging every candidate a complete singleton envelope is
    therefore conservative while using exactly the compatible serialization
    conventions that reach the agent.
    """

    channel = str(observation.get("channel"))
    row_reference = str(
        observation.get("observation_id") or logical_observation_key(observation)
    )
    table = compact_observation_fragment(
        observation,
        query_time_seconds=query_time_seconds,
        request=request,
    )
    return count_serialized_tokens(
        {
            "table_schema": _table_metadata(
                bool(request and request.detail == "raw"),
                (channel,),
            ),
            "tables": {channel: table},
            "evidence_groups": [
                {
                    "group": 0,
                    "anchor": row_reference,
                    # Each selected observation appears exactly once as an
                    # anchor or support. Treating every row as its own anchor
                    # over-reserves group structure without duplicating keys.
                    "supports": [],
                }
            ],
        },
        encoding_name=encoding_name,
    )


def _target_scope_metadata(
    target_scope_ambiguity: bool | None,
    target_scope_candidates: Sequence[Mapping[str, Any]] | None,
    *,
    retained_observation_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Validate and optionally filter the non-oracle target-scope summary."""

    if target_scope_ambiguity is None and target_scope_candidates is None:
        return {}
    if not isinstance(target_scope_ambiguity, bool):
        raise TypeError("target_scope_ambiguity must be a boolean")
    if not isinstance(target_scope_candidates, Sequence) or isinstance(
        target_scope_candidates, (str, bytes, bytearray)
    ):
        raise TypeError("target_scope_candidates must be an array")
    normalized: list[dict[str, Any]] = []
    seen_scopes: set[str] = set()
    for candidate in target_scope_candidates:
        if not isinstance(candidate, Mapping) or set(candidate) != {
            "scope",
            "estimated_role",
            "supporting_observation_ids",
        }:
            raise ValueError(
                "target_scope_candidates entries must contain exactly scope, "
                "estimated_role, and supporting_observation_ids"
            )
        scope = candidate.get("scope")
        role = candidate.get("estimated_role")
        raw_ids = candidate.get("supporting_observation_ids")
        if not isinstance(scope, str) or not scope.strip():
            raise ValueError("target scope must be a non-empty string")
        scope = scope.strip()
        if scope in seen_scopes:
            raise ValueError("target scopes must be unique")
        seen_scopes.add(scope)
        if role not in _ESTIMATED_TARGET_ROLES:
            raise ValueError(
                "estimated target role must be one of "
                f"{sorted(_ESTIMATED_TARGET_ROLES)}"
            )
        if not isinstance(raw_ids, Sequence) or isinstance(
            raw_ids, (str, bytes, bytearray)
        ):
            raise TypeError("supporting_observation_ids must be an array")
        if not raw_ids:
            raise ValueError("supporting_observation_ids must not be empty")
        validated_ids: list[str] = []
        for raw_id in raw_ids:
            if not isinstance(raw_id, str) or not raw_id.strip():
                raise ValueError("supporting observation IDs must be non-empty strings")
            validated_ids.append(raw_id.strip())
        if len(set(validated_ids)) != len(validated_ids):
            raise ValueError("supporting observation IDs must be unique")
        observation_ids: list[str] = []
        for observation_id in validated_ids:
            if (
                retained_observation_ids is None
                or observation_id in retained_observation_ids
            ):
                observation_ids.append(observation_id)
        if not observation_ids:
            continue
        normalized.append(
            {
                "scope": scope,
                "estimated_role": role,
                "supporting_observation_ids": observation_ids,
            }
        )
    return {
        # Filtering a targeted/raw view can remove the only evidence for one
        # competing scope. Do not preserve an unsupported ambiguity flag.
        "target_scope_ambiguity": bool(target_scope_ambiguity and len(normalized) >= 2),
        "target_scope_candidates": normalized,
    }


def compact_statebundle_bundle_token_cost(
    observations: Sequence[Mapping[str, Any]],
    groups: Sequence[Mapping[str, Any]],
    *,
    query_time_seconds: float,
    encoding_name: str = "o200k_base",
    request: AgentObservationRequest | None = None,
    recent_seconds: float = DEFAULT_RECENT_SECONDS,
    target_scope_ambiguity: bool | None = None,
    target_scope_candidates: Sequence[Mapping[str, Any]] | None = None,
) -> int:
    """Count one complete compact StateBundle telemetry payload exactly.

    Unlike the conservative singleton estimator above, this function renders
    all supplied observations together.  Channel-table headers, metric time
    axes, alert aggregation, and evidence-group envelopes are therefore shared
    exactly as they are in the agent-facing payload. When the paired
    ``target_scope_*`` arguments are supplied, their selector-estimated public
    summary is included in the same token count and filtered to the supplied
    observation IDs. Omitting both arguments retains compatibility with older
    ``statebundle.output.v1`` artifacts. Callers are responsible for supplying
    observations and group references that have already passed request
    filtering.
    """

    target_scope = _target_scope_metadata(
        target_scope_ambiguity,
        target_scope_candidates,
        retained_observation_ids={
            str(item.get("observation_id") or logical_observation_key(item))
            for item in observations
        },
    )
    if not observations and not groups and not target_scope:
        # B is the telemetry-evidence budget. An empty evidence set therefore
        # has zero cost under sum_i ell_i even though its transport object has
        # constant JSON structure.
        return 0

    source_request = request or AgentObservationRequest()
    tables = render_tables(
        observations,
        query_time_seconds=query_time_seconds,
        request=source_request,
        recent_seconds=recent_seconds,
    )
    payload = {
        "table_schema": _table_metadata(
            source_request.detail == "raw",
            tables,
        ),
        "tables": tables,
        "evidence_groups": [dict(group) for group in groups],
        **target_scope,
    }
    return count_serialized_tokens(
        payload,
        encoding_name=encoding_name,
    )


def selected_statebundle_observations(
    output: Mapping[str, Any],
) -> tuple[list[Mapping[str, Any]], list[dict[str, Any]]]:
    """Flatten public StateBundle groups while retaining renderer-only roles."""

    observations: list[Mapping[str, Any]] = []
    groups: list[dict[str, Any]] = []
    for group_index, group in enumerate(output.get("evidence_groups", [])):
        if not isinstance(group, Mapping):
            continue
        anchor = group.get("anchor")
        if not isinstance(anchor, Mapping):
            continue
        observations.append(anchor)
        supports = [
            item
            for item in group.get("corroborating_observations", [])
            if isinstance(item, Mapping)
        ]
        observations.extend(supports)
        groups.append(
            {
                "group": group_index,
                "anchor": str(
                    anchor.get("observation_id") or logical_observation_key(anchor)
                ),
                "supports": [
                    str(item.get("observation_id") or logical_observation_key(item))
                    for item in supports
                ],
            }
        )
    return observations, groups


def _budget_statebundle_serialization(
    observations: Sequence[Mapping[str, Any]],
    groups: Sequence[Mapping[str, Any]],
    *,
    token_budget: int,
    query_time_seconds: float,
    request: AgentObservationRequest,
    recent_seconds: float = DEFAULT_RECENT_SECONDS,
    target_scope_ambiguity: bool | None = None,
    target_scope_candidates: Sequence[Mapping[str, Any]] | None = None,
) -> tuple[
    list[Mapping[str, Any]],
    list[dict[str, Any]],
    dict[str, int],
    dict[str, Any],
]:
    """Admit selected rows using their actual requested serialization cost.

    StateBundle learned ordering is fixed before this layer.  A raw drill-down
    can be much larger than the overview representation cost used by model
    inference, so this deterministic pass preserves group/row order while
    skipping selected rows that cannot fit B.  Every admission is costed as
    the complete prospective shared-table/group payload, rather than as a sum
    of conservative singleton envelopes.  It never introduces an unselected
    observation.
    """

    by_reference = {
        str(item.get("observation_id") or logical_observation_key(item)): item
        for item in observations
    }
    admitted_references: set[str] = set()
    admitted: list[Mapping[str, Any]] = []
    admitted_groups: list[dict[str, Any]] = []
    used_tokens = 0
    considered = 0
    for group in groups:
        ordered = [group.get("anchor"), *group.get("supports", [])]
        retained: list[str] = []
        for raw_reference in ordered:
            if not isinstance(raw_reference, str):
                continue
            observation = by_reference.get(raw_reference)
            if observation is None:
                continue
            already_admitted = raw_reference in admitted_references
            if not already_admitted:
                considered += 1
            prospective_retained = [*retained, raw_reference]
            prospective_group = {
                "group": group.get("group"),
                "anchor": prospective_retained[0],
                "supports": prospective_retained[1:],
            }
            prospective_observations = (
                admitted if already_admitted else [*admitted, observation]
            )
            prospective_groups = [*admitted_groups, prospective_group]
            prospective_references = {
                str(item.get("observation_id") or logical_observation_key(item))
                for item in prospective_observations
            }
            prospective_target_scope = _target_scope_metadata(
                target_scope_ambiguity,
                target_scope_candidates,
                retained_observation_ids=prospective_references,
            )
            cost = compact_statebundle_bundle_token_cost(
                prospective_observations,
                prospective_groups,
                query_time_seconds=query_time_seconds,
                request=request,
                recent_seconds=recent_seconds,
                target_scope_ambiguity=prospective_target_scope.get(
                    "target_scope_ambiguity"
                ),
                target_scope_candidates=prospective_target_scope.get(
                    "target_scope_candidates"
                ),
            )
            if cost > token_budget:
                continue
            used_tokens = cost
            if not already_admitted:
                admitted_references.add(raw_reference)
                admitted.append(observation)
            retained.append(raw_reference)
        if retained:
            admitted_groups.append(
                {
                    "group": group.get("group"),
                    "anchor": retained[0],
                    "supports": retained[1:],
                }
            )
    final_target_scope = _target_scope_metadata(
        target_scope_ambiguity,
        target_scope_candidates,
        retained_observation_ids=admitted_references,
    )
    return (
        admitted,
        admitted_groups,
        {
            "serialization_admission_used_tokens": used_tokens,
            "serialization_admission_dropped_count": considered - len(admitted),
        },
        final_target_scope,
    )


def _counts(observations: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return {
        channel: sum(item.get("channel") == channel for item in observations)
        for channel in CANONICAL_CHANNELS
        if any(item.get("channel") == channel for item in observations)
    }


def render_snapshot(
    snapshot_or_output: Mapping[str, Any],
    *,
    request: AgentObservationRequest | None = None,
    condition: str = "full-canonical",
    token_budget: int | None = None,
    canonical_context: Mapping[str, Any] | None = None,
    recent_seconds: float = DEFAULT_RECENT_SECONDS,
) -> dict[str, Any]:
    """Render one canonical or StateBundle result as an agent snapshot."""

    if condition not in {"full-canonical", "statebundle"}:
        raise ValueError(f"unsupported observation condition: {condition}")
    request = request or AgentObservationRequest()
    source_schema = snapshot_or_output.get("schema_version")
    if source_schema == CANONICAL_SCHEMA_VERSION:
        context = snapshot_or_output
        observations = [
            item
            for item in snapshot_or_output.get("observations", [])
            if isinstance(item, Mapping)
        ]
        statebundle_groups: list[dict[str, Any]] | None = None
        target_scope: dict[str, Any] = {}
    elif source_schema == STATEBUNDLE_OUTPUT_SCHEMA_VERSION:
        context = canonical_context or snapshot_or_output
        observations, statebundle_groups = selected_statebundle_observations(
            snapshot_or_output
        )
        has_target_scope = (
            "target_scope_ambiguity" in snapshot_or_output
            or "target_scope_candidates" in snapshot_or_output
        )
        target_scope = (
            _target_scope_metadata(
                snapshot_or_output.get("target_scope_ambiguity"),
                snapshot_or_output.get("target_scope_candidates"),
            )
            if has_target_scope
            else {}
        )
    else:
        raise ValueError(
            f"unsupported telemetry schema for compact rendering: {source_schema!r}"
        )

    query_time = float(
        context.get(
            "query_time_seconds", snapshot_or_output.get("query_time_seconds", 0.0)
        )
    )
    observations = filter_observations(
        observations,
        query_time_seconds=query_time,
        request=request,
    )
    if statebundle_groups is not None:
        retained_references = {
            str(item.get("observation_id") or logical_observation_key(item))
            for item in observations
        }
        filtered_groups: list[dict[str, Any]] = []
        for group in statebundle_groups:
            ordered_keys = [group.get("anchor"), *group.get("supports", [])]
            retained = [
                str(key)
                for key in ordered_keys
                if isinstance(key, str) and key in retained_references
            ]
            if retained:
                filtered_groups.append(
                    {
                        "group": group.get("group"),
                        "anchor": retained[0],
                        "supports": retained[1:],
                    }
                )
        statebundle_groups = filtered_groups
        configured_budget = int(
            snapshot_or_output.get("token_budget", token_budget or 1)
        )
        observations, statebundle_groups, serialization_admission, target_scope = (
            _budget_statebundle_serialization(
                observations,
                statebundle_groups,
                token_budget=configured_budget,
                query_time_seconds=query_time,
                request=request,
                recent_seconds=recent_seconds,
                target_scope_ambiguity=target_scope.get("target_scope_ambiguity"),
                target_scope_candidates=target_scope.get("target_scope_candidates"),
            )
        )
    else:
        serialization_admission = None
    tables = render_tables(
        observations,
        query_time_seconds=query_time,
        request=request,
        recent_seconds=recent_seconds,
    )
    result: dict[str, Any] = {
        "schema_version": AGENT_SCHEMA_VERSION,
        "source_schema_version": source_schema,
        "condition": condition,
        "mode": "snapshot",
        "episode_id": context.get("episode_id", snapshot_or_output.get("incident_id")),
        "snapshot_id": snapshot_or_output.get("snapshot_id")
        or context.get("snapshot_id"),
        "causal_cut": context.get("query_watermark_sequence"),
        "query_time_seconds": query_time,
        "window": [max(0.0, query_time - request.lookback_seconds), query_time],
        "requested_channels": list(request.requested_channels),
        # One count map is sufficient.  In budgeted conditions this reports
        # only evidence actually serialized, avoiding a full-snapshot count
        # side channel and a duplicate field in Full-Canonical.
        "channel_counts": _counts(observations),
        "table_schema": _table_metadata(request.detail == "raw", tables),
        "tables": tables,
    }
    if result["causal_cut"] is None:
        result.pop("causal_cut")
    if source_schema == STATEBUNDLE_OUTPUT_SCHEMA_VERSION:
        actual = compact_statebundle_bundle_token_cost(
            observations,
            statebundle_groups,
            query_time_seconds=query_time,
            request=request,
            recent_seconds=recent_seconds,
            target_scope_ambiguity=target_scope.get("target_scope_ambiguity"),
            target_scope_candidates=target_scope.get("target_scope_candidates"),
        )
        configured_budget = int(
            snapshot_or_output.get("token_budget", token_budget or actual)
        )
        within_budget = actual <= configured_budget
        result["budget"] = {
            "token_budget": configured_budget,
            "actual_serialized_tokens": actual,
            "within_budget": within_budget,
            "model_estimated_tokens": snapshot_or_output.get("used_tokens"),
            "tokenizer": "tiktoken:o200k_base",
            **(serialization_admission or {}),
        }
        result["evidence_groups"] = statebundle_groups
        result.update(deepcopy(target_scope))
        request_status = snapshot_or_output.get("request_status")
        if isinstance(request_status, Mapping):
            # Surface the selector's safe, aggregate drill-down status so an
            # oversized or empty request can be narrowed instead of silently
            # substituting unrelated telemetry.
            result["request_status"] = deepcopy(dict(request_status))
        if condition in {"statebundle"} and not within_budget:
            raise RuntimeError(
                "StateBundle evidence exceeds the actual compact agent token budget"
            )
    return _drop_none(result)


def _drop_none(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _drop_none(item) for key, item in value.items() if item is not None
        }
    if isinstance(value, list):
        return [_drop_none(item) for item in value]
    return value


def _observation_groups(
    observations: Sequence[Mapping[str, Any]],
) -> dict[str, list[Mapping[str, Any]]]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for observation in observations:
        groups[logical_observation_key(observation)].append(observation)
    for items in groups.values():
        items.sort(
            key=lambda item: (_event_end(item), str(item.get("observation_id", "")))
        )
    return dict(groups)


def _quality_signature(observation: Mapping[str, Any]) -> str:
    metadata = observation.get("metadata")
    metadata = metadata if isinstance(metadata, Mapping) else {}
    return deterministic_json(_compact_quality(metadata))


def _agent_observation_signature(
    observation: Mapping[str, Any],
    *,
    query_time_seconds: float,
    request: AgentObservationRequest,
    recent_seconds: float,
) -> str:
    """Fingerprint only information visible in the deterministic projection.

    Canonical cut/provenance fields can legitimately change while an
    observation's agent-visible meaning does not.  Comparing the projected
    singleton prevents those internal changes from producing false updates.
    """

    return deterministic_json(
        render_tables(
            [observation],
            query_time_seconds=query_time_seconds,
            request=request,
            recent_seconds=recent_seconds,
        )
    )


@dataclass(slots=True)
class _PreviousView:
    snapshot_id: str | None
    query_time_seconds: float
    observations: list[Mapping[str, Any]]
    universe_keys: frozenset[str]
    seen_schema_channels: frozenset[str]


class AgentTelemetryRenderer:
    """Stateful snapshot/delta renderer for one benchmark episode."""

    def __init__(
        self,
        *,
        condition: str = "full-canonical",
        token_budget: int | None = None,
        recent_seconds: float = DEFAULT_RECENT_SECONDS,
    ) -> None:
        self.condition = condition
        if self.condition not in {"full-canonical", "statebundle"}:
            raise ValueError(f"unsupported observation condition: {condition}")
        self.token_budget = token_budget
        self.recent_seconds = recent_seconds
        self._previous: dict[tuple[Any, ...], _PreviousView] = {}
        self.audit_records: list[dict[str, Any]] = []
        # Trusted in-memory snapshots support deltas and never enter prompts or
        # result artifacts.  Retaining these does not change the canonical store.
        self._latest_canonical_snapshot: Mapping[str, Any] | None = None

    @property
    def latest_canonical_snapshot(self) -> Mapping[str, Any] | None:
        return self._latest_canonical_snapshot

    def render(
        self,
        snapshot_or_output: Mapping[str, Any],
        *,
        request: AgentObservationRequest | None = None,
        canonical_context: Mapping[str, Any] | None = None,
        initial: bool = False,
    ) -> dict[str, Any]:
        request = request or AgentObservationRequest()
        if (
            canonical_context is not None
            and canonical_context.get("schema_version") == CANONICAL_SCHEMA_VERSION
        ):
            self._latest_canonical_snapshot = deepcopy(canonical_context)
        elif snapshot_or_output.get("schema_version") == CANONICAL_SCHEMA_VERSION:
            self._latest_canonical_snapshot = deepcopy(snapshot_or_output)
        full = render_snapshot(
            snapshot_or_output,
            request=request,
            condition=self.condition,
            token_budget=self.token_budget,
            canonical_context=canonical_context,
            recent_seconds=self.recent_seconds,
        )
        source_schema = snapshot_or_output.get("schema_version")
        if source_schema == CANONICAL_SCHEMA_VERSION:
            raw_observations = [
                item
                for item in snapshot_or_output.get("observations", [])
                if isinstance(item, Mapping)
            ]
            raw_statebundle_groups: list[dict[str, Any]] | None = None
        else:
            raw_observations, raw_statebundle_groups = (
                selected_statebundle_observations(snapshot_or_output)
            )
        query_time = float(full.get("query_time_seconds", 0.0))
        current = filter_observations(
            raw_observations,
            query_time_seconds=query_time,
            request=request,
        )
        universe = current
        if (
            source_schema == STATEBUNDLE_OUTPUT_SCHEMA_VERSION
            and isinstance(canonical_context, Mapping)
            and canonical_context.get("schema_version") == CANONICAL_SCHEMA_VERSION
        ):
            universe = filter_observations(
                [
                    item
                    for item in canonical_context.get("observations", [])
                    if isinstance(item, Mapping)
                ],
                query_time_seconds=query_time,
                request=request,
            )
        if raw_statebundle_groups is not None:
            current_references = {
                str(item.get("observation_id") or logical_observation_key(item))
                for item in current
            }
            current_groups: list[dict[str, Any]] = []
            for group in raw_statebundle_groups:
                ordered = [group.get("anchor"), *group.get("supports", [])]
                retained = [
                    str(reference)
                    for reference in ordered
                    if isinstance(reference, str) and reference in current_references
                ]
                if retained:
                    current_groups.append(
                        {
                            "group": group.get("group"),
                            "anchor": retained[0],
                            "supports": retained[1:],
                        }
                    )
            current, _, _, _ = _budget_statebundle_serialization(
                current,
                current_groups,
                token_budget=int(snapshot_or_output.get("token_budget", 1)),
                query_time_seconds=query_time,
                request=request,
                recent_seconds=self.recent_seconds,
                target_scope_ambiguity=snapshot_or_output.get("target_scope_ambiguity"),
                target_scope_candidates=snapshot_or_output.get(
                    "target_scope_candidates"
                ),
            )
        view_key = request.view_key
        previous = self._previous.get(view_key)
        current_schema_channels = frozenset(
            str(item.get("channel"))
            for item in current
            if item.get("channel") in {"metric", "alert"}
        )
        self._previous[view_key] = _PreviousView(
            snapshot_id=full.get("snapshot_id"),
            query_time_seconds=query_time,
            observations=deepcopy(current),
            universe_keys=frozenset(logical_observation_key(item) for item in universe),
            seen_schema_channels=(
                current_schema_channels
                if previous is None
                else previous.seen_schema_channels | current_schema_channels
            ),
        )
        # Raw drill-down is deliberately self-contained and must not perturb or
        # depend on the overview delta chain.
        if initial or request.detail == "raw" or previous is None:
            rendered = full
        else:
            rendered = self._render_delta(
                full,
                previous=previous,
                current=current,
                current_universe_keys=frozenset(
                    logical_observation_key(item) for item in universe
                ),
                request=request,
            )
        self._record_audit(
            snapshot_or_output=snapshot_or_output,
            canonical_context=canonical_context,
            request=request,
            source_observation_count=len(raw_observations),
            represented=current,
            rendered=rendered,
        )
        return rendered

    def _record_audit(
        self,
        *,
        snapshot_or_output: Mapping[str, Any],
        canonical_context: Mapping[str, Any] | None,
        request: AgentObservationRequest,
        source_observation_count: int,
        represented: Sequence[Mapping[str, Any]],
        rendered: Mapping[str, Any],
    ) -> None:
        """Retain hashes/counts for reproducibility, never prompt telemetry."""

        canonical = (
            canonical_context
            if isinstance(canonical_context, Mapping)
            and canonical_context.get("schema_version") == CANONICAL_SCHEMA_VERSION
            else (
                snapshot_or_output
                if snapshot_or_output.get("schema_version") == CANONICAL_SCHEMA_VERSION
                else None
            )
        )
        canonical_observations = (
            canonical.get("observations", []) if isinstance(canonical, Mapping) else []
        )
        represented_ids = sorted(
            str(item.get("observation_id", "")) for item in represented
        )
        self.audit_records.append(
            {
                "call_index": len(self.audit_records) + 1,
                "condition": self.condition,
                "input_schema_version": snapshot_or_output.get("schema_version"),
                "input_snapshot_id": snapshot_or_output.get("snapshot_id"),
                "canonical_snapshot_id": (
                    canonical.get("snapshot_id")
                    if isinstance(canonical, Mapping)
                    else None
                ),
                "canonical_observation_count": (
                    len(canonical_observations)
                    if isinstance(canonical_observations, Sequence)
                    and not isinstance(canonical_observations, (str, bytes))
                    else None
                ),
                "canonical_snapshot_sha256": (
                    hashlib.sha256(
                        deterministic_json(canonical).encode("utf-8")
                    ).hexdigest()
                    if isinstance(canonical, Mapping)
                    else None
                ),
                "source_observation_count": source_observation_count,
                "represented_observation_count": len(represented),
                "represented_observation_ids_sha256": hashlib.sha256(
                    deterministic_json(represented_ids).encode("utf-8")
                ).hexdigest(),
                "request": {
                    "include_config": request.include_config,
                    "channels": list(request.requested_channels),
                    "lookback_seconds": request.lookback_seconds,
                    "log_limit": request.log_limit,
                    "detail": request.detail,
                    "metric_names": list(request.metric_names),
                    "entity_ids": list(request.entity_ids),
                    "subsystem_ids": list(request.subsystem_ids),
                    "alert_names": list(request.alert_names),
                },
                "rendered_schema_version": rendered.get("schema_version"),
                "rendered_mode": rendered.get("mode"),
                "rendered_tokens": count_serialized_tokens(rendered),
                "tokenizer": "tiktoken:o200k_base",
                "rendered_sha256": hashlib.sha256(
                    deterministic_json(rendered).encode("utf-8")
                ).hexdigest(),
            }
        )

    def _render_delta(
        self,
        full: Mapping[str, Any],
        *,
        previous: _PreviousView,
        current: Sequence[Mapping[str, Any]],
        current_universe_keys: frozenset[str],
        request: AgentObservationRequest,
    ) -> dict[str, Any]:
        previous_groups = _observation_groups(previous.observations)
        current_groups = _observation_groups(current)
        added: list[Mapping[str, Any]] = []
        updated: list[Mapping[str, Any]] = []
        previous_for_updates: dict[str, Mapping[str, Any]] = {}
        removed: list[dict[str, Any]] = []
        quality_changes: list[dict[str, Any]] = []
        selection_admitted: list[str] = []
        selection_evicted: list[str] = []
        budgeted = self.condition == "statebundle"
        for key in sorted(set(previous_groups) | set(current_groups)):
            before = previous_groups.get(key, [])
            after = current_groups.get(key, [])
            if not before:
                added.extend(after)
                if budgeted and key in previous.universe_keys:
                    selection_admitted.append(key)
                continue
            if not after:
                if budgeted and key in current_universe_keys:
                    selection_evicted.append(key)
                    continue
                removed.append(
                    {
                        "key": key,
                        "channel": before[-1].get("channel"),
                        "observation_ids": [
                            item.get("observation_id") for item in before
                        ],
                    }
                )
                continue
            before_by_id = {str(item.get("observation_id")): item for item in before}
            after_by_id = {str(item.get("observation_id")): item for item in after}
            new_ids = sorted(set(after_by_id) - set(before_by_id))
            removed_ids = sorted(set(before_by_id) - set(after_by_id))
            # A logical singleton with a new content-derived canonical ID is an
            # update (metrics/log aggregates/alerts), not add+remove churn.
            if len(before) == len(after) == 1 and new_ids and removed_ids:
                updated.append(after[0])
                previous_for_updates[key] = before[0]
                if _quality_signature(before[0]) != _quality_signature(after[0]):
                    quality_changes.append(
                        {
                            "key": key,
                            "before": _compact_quality(
                                before[0].get("metadata", {})
                                if isinstance(before[0].get("metadata"), Mapping)
                                else {}
                            ),
                            "after": _compact_quality(
                                after[0].get("metadata", {})
                                if isinstance(after[0].get("metadata"), Mapping)
                                else {}
                            ),
                        }
                    )
                continue
            added.extend(after_by_id[item_id] for item_id in new_ids)
            if removed_ids:
                removed.append(
                    {
                        "key": key,
                        "channel": before[-1].get("channel"),
                        "observation_ids": removed_ids,
                    }
                )
            for item_id in sorted(set(before_by_id) & set(after_by_id)):
                if _agent_observation_signature(
                    before_by_id[item_id],
                    query_time_seconds=previous.query_time_seconds,
                    request=request,
                    recent_seconds=self.recent_seconds,
                ) != _agent_observation_signature(
                    after_by_id[item_id],
                    query_time_seconds=float(full.get("query_time_seconds", 0.0)),
                    request=request,
                    recent_seconds=self.recent_seconds,
                ):
                    updated.append(after_by_id[item_id])
                    previous_for_updates[key] = before_by_id[item_id]
                if _quality_signature(before_by_id[item_id]) != _quality_signature(
                    after_by_id[item_id]
                ):
                    quality_changes.append(
                        {
                            "key": key,
                            "before": _compact_quality(
                                before_by_id[item_id].get("metadata", {})
                            ),
                            "after": _compact_quality(
                                after_by_id[item_id].get("metadata", {})
                            ),
                        }
                    )

        query_time = float(full.get("query_time_seconds", 0.0))
        added_tables = render_tables(
            added,
            query_time_seconds=query_time,
            request=request,
            recent_seconds=self.recent_seconds,
        )
        updated_tables = render_tables(
            updated,
            query_time_seconds=query_time,
            request=request,
            recent_seconds=self.recent_seconds,
            previous_by_key=previous_for_updates,
        )
        delta_channels = set(added_tables) | set(updated_tables)
        newly_needed_schema = delta_channels - set(previous.seen_schema_channels)
        delta = {
            "schema_version": AGENT_DELTA_SCHEMA_VERSION,
            "mode": "delta",
            "base_snapshot_id": previous.snapshot_id,
            "snapshot_id": full.get("snapshot_id"),
            "causal_cut": full.get("causal_cut"),
            "query_time_seconds": query_time,
            "added": added_tables,
            "updated": updated_tables,
            "removed": removed,
            "quality_changes": quality_changes,
            "channel_counts": full.get("channel_counts"),
        }
        table_schema = _table_metadata(
            request.detail == "raw",
            newly_needed_schema,
        )
        if table_schema:
            delta["table_schema"] = table_schema
        if "budget" in full:
            delta["budget"] = full["budget"]
        if "evidence_groups" in full:
            delta["evidence_groups"] = full["evidence_groups"]
        if "request_status" in full:
            delta["request_status"] = deepcopy(full["request_status"])
        if "target_scope_ambiguity" in full:
            delta["target_scope_ambiguity"] = full["target_scope_ambiguity"]
        if "target_scope_candidates" in full:
            delta["target_scope_candidates"] = deepcopy(full["target_scope_candidates"])
        if selection_admitted or selection_evicted:
            delta["selection_changes"] = {
                "admitted": selection_admitted,
                "evicted": selection_evicted,
            }
        return _drop_none(delta)


__all__ = [
    "AGENT_DELTA_SCHEMA_VERSION",
    "AGENT_SCHEMA_VERSION",
    "AgentObservationRequest",
    "AgentTelemetryRenderer",
    "CANONICAL_CHANNELS",
    "compact_observation_fragment",
    "compact_statebundle_bundle_token_cost",
    "compact_statebundle_observation_token_cost",
    "count_serialized_tokens",
    "deterministic_json",
    "estimate_tokens",
    "filter_observations",
    "logical_observation_key",
    "render_snapshot",
    "render_tables",
    "selected_statebundle_observations",
]
