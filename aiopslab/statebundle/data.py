"""Parse live canonical snapshots and enforce causal/oracle isolation."""

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any
import math
from .types import CanonicalIncident

_ADDITIONAL_FORBIDDEN_OBSERVABLE_KEYS = frozenset(
    {
        "inference_visible_target",
        "label_provenance",
        "observable_evidence",
    }
)


class StateBundleDataError(ValueError):
    """Raised when a canonical snapshot violates the StateBundle data contract."""


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise StateBundleDataError(f"{field} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise StateBundleDataError(f"{field} must be a finite number")
    return result


def _nonempty_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise StateBundleDataError(f"{field} must be non-empty text")
    return value.strip()


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise StateBundleDataError(f"{field} must be an object")
    if not all(isinstance(key, str) for key in value):
        raise StateBundleDataError(f"{field} must use string keys")
    return value


def _array(value: Any, field: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise StateBundleDataError(f"{field} must be an array")
    return value


def _reject_additional_hidden_fields(value: Any, path: str = "observation") -> None:
    """Close gaps intentionally left by the general-purpose typed contract."""

    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).strip().lower()
            if normalized in _ADDITIONAL_FORBIDDEN_OBSERVABLE_KEYS:
                raise StateBundleDataError(
                    f"training-only field is not allowed at {path}.{key}"
                )
            _reject_additional_hidden_fields(item, f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            _reject_additional_hidden_fields(item, f"{path}[{index}]")


def parse_canonical_snapshot(snapshot: Mapping[str, Any]) -> CanonicalIncident:
    """Parse one live, inference-safe canonical snapshot.

    Every observation must have been available at the query time.  When the
    source provides an opaque query watermark, observations must also belong
    to that same causal cut; timestamp-only production sources may omit it.
    This function accepts no label sidecar and
    ``CanonicalIncident.from_dict`` rejects recognizable hidden-supervision
    keys recursively.
    """

    raw = _mapping(snapshot, "canonical snapshot")
    _reject_additional_hidden_fields(raw, "snapshot")
    marker = raw.get("query_watermark_sequence")
    if marker is not None and (
        isinstance(marker, bool) or not isinstance(marker, (str, int))
    ):
        raise StateBundleDataError(
            "query_watermark_sequence must be a non-empty string, integer, or null"
        )
    if isinstance(marker, str) and not marker.strip():
        raise StateBundleDataError(
            "query_watermark_sequence must be a non-empty string or integer"
        )

    try:
        incident = CanonicalIncident.from_dict(raw)
    except (KeyError, TypeError, ValueError) as error:
        raise StateBundleDataError(str(error)) from error

    snapshot_window = raw.get("window")
    if snapshot_window is not None:
        window = _mapping(snapshot_window, "canonical snapshot window")
        start = _finite_number(
            window.get("start_time_seconds"), "snapshot.start_time_seconds"
        )
        end = _finite_number(
            window.get("end_time_seconds"), "snapshot.end_time_seconds"
        )
        if (
            start < 0.0
            or end < start
            or not math.isclose(
                end, incident.query_time_seconds, rel_tol=0.0, abs_tol=1e-6
            )
        ):
            raise StateBundleDataError("snapshot window must end at query_time_seconds")
        if (
            window.get("start_inclusive") is not True
            or window.get("end_inclusive") is not True
        ):
            raise StateBundleDataError("canonical snapshot window must be closed")

    declared_channels = raw.get("channels")
    if declared_channels is not None:
        channels = tuple(
            _nonempty_text(channel, "snapshot channel")
            for channel in _array(declared_channels, "snapshot channels")
        )
        observed_counts: dict[str, int] = defaultdict(int)
        for observation in incident.observations:
            observed_counts[observation.channel.value] += 1
        undeclared = sorted(set(observed_counts) - set(channels))
        if undeclared:
            raise StateBundleDataError(
                f"snapshot contains undeclared channels: {undeclared}"
            )
        raw_counts = raw.get("channel_counts")
        if raw_counts is not None:
            counts = _mapping(raw_counts, "channel_counts")
            expected = {
                channel: observed_counts.get(channel, 0) for channel in channels
            }
            if dict(counts) != expected:
                raise StateBundleDataError(
                    "channel_counts do not match canonical observations"
                )
    return incident
