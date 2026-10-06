"""Pure mitigation metrics for Data Center Twin evaluator trajectories.

This module operates only on privileged evaluator-health samples retained in
episode artifacts.  It intentionally has no dependency on agent-visible
telemetry or on the provider/agent harness.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import Any


SLO_NORMALIZATION_EPSILON = 1e-12

UPPER_BOUND = "upper_bound"
LOWER_BOUND = "lower_bound"
CONSTRAINT_DIRECTIONS = frozenset({UPPER_BOUND, LOWER_BOUND})


class MitigationMetricError(ValueError):
    """Raised when a mitigation trajectory cannot be audited safely."""


def make_constraint(
    identifier: str,
    observed_value: Any,
    threshold: Any,
    direction: str,
    source_field: str | None = None,
    raw_observed_value: Any = None,
) -> dict[str, Any]:
    """Build one auditable evaluator-health constraint record.

    Signed violations follow the manuscript convention: healthy values are at
    or below zero.  Equality-like or other Boolean evaluator predicates can be
    represented as a numeric indicator with a lower bound of one while their
    original value is retained in ``raw_observed_value``.
    """

    constraint_id = _nonempty_string(identifier, "constraint identifier")
    if source_field is not None:
        _nonempty_string(source_field, "constraint source_field")
    observed = _finite_number(observed_value, f"{constraint_id}.observed_value")
    numeric_threshold = _finite_number(threshold, f"{constraint_id}.threshold")
    normalized = _normalized_signed_violation(
        observed,
        numeric_threshold,
        direction,
        SLO_NORMALIZATION_EPSILON,
    )
    denominator = max(abs(numeric_threshold), SLO_NORMALIZATION_EPSILON)
    return {
        "identifier": constraint_id,
        "source_field": source_field,
        "observed_value": observed,
        "raw_observed_value": raw_observed_value,
        "threshold": numeric_threshold,
        "direction": direction,
        "normalization_denominator": denominator,
        "normalized_signed_violation": normalized,
        "positive_violation": max(normalized, 0.0),
        "healthy": normalized <= 0.0,
    }


def compute_slo_metrics(
    trajectory: Sequence[Mapping[str, Any]],
    fault_injection_time: Any,
    episode_horizon: Any,
    epsilon: Any = SLO_NORMALIZATION_EPSILON,
) -> dict[str, Any]:
    """Compute normalized SLO area and union violation duration.

    Each sample is treated as holding on the half-open interval beginning at
    its evaluator timestamp and ending at the next evaluator timestamp.  The
    actual interval length is used, and every interval is clipped to the
    post-fault episode range ``[fault_injection_time, episode_horizon]``.  No
    state is extrapolated beyond the final recorded sample.
    """

    fault_time, horizon = _validated_episode_bounds(
        fault_injection_time,
        episode_horizon,
    )
    numeric_epsilon = _positive_finite_number(epsilon, "epsilon")
    samples = _validated_samples(trajectory, require_health_condition=True)

    if not samples:
        return {
            "slo_violation_area": 0.0,
            "raw_slo_violation_duration": 0.0,
            "constraint_count": 0,
            "normalization_epsilon": numeric_epsilon,
            "integration_rule": "left_hold_actual_interval",
            "integrated_post_fault_time": 0.0,
        }

    expected_ids: frozenset[str] | None = None
    expected_definitions: dict[str, tuple[float, str, str | None]] = {}
    constraint_maps: list[dict[str, Mapping[str, Any]]] = []
    for index, sample in enumerate(samples):
        constraints = sample.get("constraints")
        if not isinstance(constraints, Sequence) or isinstance(
            constraints, (str, bytes, bytearray)
        ):
            raise MitigationMetricError(
                f"trajectory sample {index}.constraints must be a sequence"
            )
        by_id: dict[str, Mapping[str, Any]] = {}
        for constraint_index, constraint in enumerate(constraints):
            if not isinstance(constraint, Mapping):
                raise MitigationMetricError(
                    "trajectory sample "
                    f"{index}.constraints[{constraint_index}] must be an object"
                )
            identifier = _nonempty_string(
                constraint.get("identifier"),
                f"trajectory sample {index} constraint identifier",
            )
            if identifier in by_id:
                raise MitigationMetricError(
                    f"trajectory sample {index} repeats constraint {identifier!r}"
                )
            # Validate every numeric input even when the corresponding sample
            # has a zero-duration interval.  This keeps the saved artifact
            # independently auditable.
            observed = _finite_number(
                constraint.get("observed_value"),
                f"trajectory sample {index} {identifier}.observed_value",
            )
            threshold = _finite_number(
                constraint.get("threshold"),
                f"trajectory sample {index} {identifier}.threshold",
            )
            direction = _validated_direction(constraint.get("direction"))
            source_field = constraint.get("source_field")
            if source_field is not None:
                source_field = _nonempty_string(
                    source_field,
                    f"trajectory sample {index} {identifier}.source_field",
                )
            definition = (threshold, direction, source_field)
            if identifier not in expected_definitions:
                expected_definitions[identifier] = definition
            elif definition != expected_definitions[identifier]:
                raise MitigationMetricError(
                    "health constraint definition changed at trajectory sample "
                    f"{index} for {identifier!r}: "
                    f"expected={expected_definitions[identifier]!r}, observed={definition!r}"
                )

            recomputed = _normalized_signed_violation(
                observed,
                threshold,
                direction,
                numeric_epsilon,
            )
            stored_violation = _finite_number(
                constraint.get("normalized_signed_violation"),
                f"trajectory sample {index} "
                f"{identifier}.normalized_signed_violation",
            )
            if not math.isclose(
                stored_violation,
                recomputed,
                rel_tol=1e-12,
                abs_tol=numeric_epsilon,
            ):
                raise MitigationMetricError(
                    "saved normalized_signed_violation is inconsistent with "
                    f"constraint inputs at trajectory sample {index} for {identifier!r}"
                )
            stored_healthy = constraint.get("healthy")
            if type(stored_healthy) is not bool or stored_healthy != (recomputed <= 0.0):
                raise MitigationMetricError(
                    "saved healthy state is inconsistent with constraint inputs "
                    f"at trajectory sample {index} for {identifier!r}"
                )
            stored_denominator = _finite_number(
                constraint.get("normalization_denominator"),
                f"trajectory sample {index} "
                f"{identifier}.normalization_denominator",
            )
            if not math.isclose(
                stored_denominator,
                max(abs(threshold), numeric_epsilon),
                rel_tol=1e-12,
                abs_tol=numeric_epsilon,
            ):
                raise MitigationMetricError(
                    "saved normalization_denominator is inconsistent with "
                    f"constraint inputs at trajectory sample {index} for {identifier!r}"
                )
            by_id[identifier] = constraint

        current_ids = frozenset(by_id)
        if expected_ids is None:
            expected_ids = current_ids
        elif current_ids != expected_ids:
            missing = sorted(expected_ids - current_ids)
            added = sorted(current_ids - expected_ids)
            raise MitigationMetricError(
                "health constraint identifiers changed at trajectory sample "
                f"{index}: missing={missing}, added={added}"
            )
        if sample["health_condition"] != all(
            constraint.get("healthy") is True for constraint in by_id.values()
        ):
            raise MitigationMetricError(
                "saved health_condition is inconsistent with the conjunction "
                f"of constraints at trajectory sample {index}"
            )
        constraint_maps.append(by_id)

    constraint_count = len(expected_ids or ())
    if constraint_count == 0 and len(samples) > 1:
        raise MitigationMetricError(
            "SLO area is undefined for a trajectory with no health constraints"
        )

    slo_area_numerator = 0.0
    raw_violation_duration = 0.0
    integrated_time = 0.0
    for index in range(len(samples) - 1):
        interval_start = max(samples[index]["simulator_time"], fault_time)
        interval_end = min(samples[index + 1]["simulator_time"], horizon)
        delta_t = max(0.0, interval_end - interval_start)
        if delta_t == 0.0:
            continue

        positive_violations: list[float] = []
        for identifier in sorted(expected_ids or ()):
            constraint = constraint_maps[index][identifier]
            violation = _normalized_signed_violation(
                _finite_number(
                    constraint.get("observed_value"),
                    f"trajectory sample {index} {identifier}.observed_value",
                ),
                _finite_number(
                    constraint.get("threshold"),
                    f"trajectory sample {index} {identifier}.threshold",
                ),
                constraint.get("direction"),
                numeric_epsilon,
            )
            positive_violations.append(max(violation, 0.0))

        slo_area_numerator += sum(positive_violations) * delta_t
        if any(violation > 0.0 for violation in positive_violations):
            # Duration is the union of unhealthy intervals, not a sum over
            # constraints, so simultaneous violations count only once.
            raw_violation_duration += delta_t
        integrated_time += delta_t

    slo_area = slo_area_numerator / constraint_count if constraint_count else 0.0
    return {
        "slo_violation_area": slo_area,
        "raw_slo_violation_duration": raw_violation_duration,
        "constraint_count": constraint_count,
        "normalization_epsilon": numeric_epsilon,
        "integration_rule": "left_hold_actual_interval",
        "integrated_post_fault_time": integrated_time,
    }


def compute_stable_recovery(
    trajectory: Sequence[Mapping[str, Any]],
    fault_injection_time: Any,
    episode_horizon: Any,
    stability_window: Any,
) -> dict[str, Any]:
    """Find the earliest fully verifiable post-fault stable-recovery window."""

    fault_time, horizon = _validated_episode_bounds(
        fault_injection_time,
        episode_horizon,
    )
    window = _nonnegative_finite_number(stability_window, "stability_window")
    samples = _validated_samples(trajectory, require_health_condition=True)

    recovery_start: float | None = None
    verification_time: float | None = None
    for candidate_index, candidate in enumerate(samples):
        candidate_time = candidate["simulator_time"]
        if candidate_time < fault_time or candidate_time > horizon:
            continue
        if candidate["health_condition"] is not True:
            continue

        required_end = candidate_time + window
        if required_end > horizon:
            # The complete stability window cannot fit inside this episode.
            continue

        coverage_index = _first_sample_at_or_after(
            samples,
            candidate_index,
            required_end,
            horizon,
        )
        if coverage_index is None:
            # A healthy candidate near the recorded end cannot be credited
            # without evidence covering its complete stability window.
            continue
        coverage_end_index = coverage_index
        if samples[coverage_index]["simulator_time"] == required_end:
            # The stability interval is closed at t+H. Ordered transitions at
            # that exact simulator timestamp are all part of the endpoint;
            # accepting only the first could hide a same-time relapse.
            while (
                coverage_end_index + 1 < len(samples)
                and samples[coverage_end_index + 1]["simulator_time"]
                == required_end
            ):
                coverage_end_index += 1
        if all(
            sample["health_condition"] is True
            for sample in samples[candidate_index : coverage_end_index + 1]
        ):
            recovery_start = candidate_time
            verification_time = samples[coverage_end_index]["simulator_time"]
            break

    remaining_horizon = horizon - fault_time
    if recovery_start is None:
        return {
            "stable_recovery_succeeded": False,
            "recovery_start_time": None,
            "recovery_verification_time": None,
            "time_to_stable_recovery": None,
            "penalized_time_to_stable_recovery": remaining_horizon,
        }

    recovery_duration = recovery_start - fault_time
    return {
        "stable_recovery_succeeded": True,
        "recovery_start_time": recovery_start,
        "recovery_verification_time": verification_time,
        "time_to_stable_recovery": recovery_duration,
        "penalized_time_to_stable_recovery": min(
            recovery_duration,
            remaining_horizon,
        ),
    }


def annotate_stable_recovery(
    trajectory: Sequence[Mapping[str, Any]],
    result: Mapping[str, Any],
    stability_window: Any,
) -> list[dict[str, Any]]:
    """Return trajectory copies annotated with recovery-achievement state.

    Recovery is considered achieved at the first retained evaluator sample
    that verifies the complete window, rather than retrospectively at the
    candidate recovery-start sample.
    """

    window = _nonnegative_finite_number(stability_window, "stability_window")
    samples = _validated_samples(trajectory, require_health_condition=False)
    succeeded = result.get("stable_recovery_succeeded") is True
    recovery_start = result.get("recovery_start_time")
    verification_time = result.get("recovery_verification_time")

    if succeeded:
        numeric_start = _finite_number(recovery_start, "recovery_start_time")
        if verification_time is None:
            achievement_time = numeric_start + window
        else:
            achievement_time = _finite_number(
                verification_time,
                "recovery_verification_time",
            )
            if achievement_time < numeric_start + window:
                raise MitigationMetricError(
                    "recovery_verification_time precedes the full stability window"
                )
    else:
        achievement_time = math.inf

    return [
        {
            **dict(sample),
            "stable_recovery_achieved": bool(
                succeeded and sample["simulator_time"] >= achievement_time
            ),
        }
        for sample in samples
    ]


def _validated_samples(
    trajectory: Sequence[Mapping[str, Any]],
    *,
    require_health_condition: bool,
) -> list[dict[str, Any]]:
    if not isinstance(trajectory, Sequence) or isinstance(
        trajectory, (str, bytes, bytearray)
    ):
        raise MitigationMetricError("trajectory must be a sequence")

    samples: list[dict[str, Any]] = []
    previous_time: float | None = None
    for index, raw_sample in enumerate(trajectory):
        if not isinstance(raw_sample, Mapping):
            raise MitigationMetricError(f"trajectory sample {index} must be an object")
        sample = dict(raw_sample)
        sample_time = _finite_number(
            sample.get("simulator_time"),
            f"trajectory sample {index}.simulator_time",
        )
        if previous_time is not None and sample_time < previous_time:
            raise MitigationMetricError(
                "trajectory simulator timestamps must be monotonically "
                f"nondecreasing: sample {index} has {sample_time} after {previous_time}"
            )
        if (
            require_health_condition
            and type(sample.get("health_condition")) is not bool
        ):
            raise MitigationMetricError(
                f"trajectory sample {index}.health_condition must be a boolean"
            )
        sample["simulator_time"] = sample_time
        samples.append(sample)
        previous_time = sample_time
    return samples


def _first_sample_at_or_after(
    samples: Sequence[Mapping[str, Any]],
    start_index: int,
    target_time: float,
    horizon: float,
) -> int | None:
    for index in range(start_index, len(samples)):
        sample_time = samples[index]["simulator_time"]
        if sample_time > horizon:
            return None
        if sample_time >= target_time:
            return index
    return None


def _normalized_signed_violation(
    observed_value: float,
    threshold: float,
    direction: Any,
    epsilon: float,
) -> float:
    validated_direction = _validated_direction(direction)
    denominator = max(abs(threshold), epsilon)
    if validated_direction == UPPER_BOUND:
        normalized = (observed_value - threshold) / denominator
    else:
        normalized = (threshold - observed_value) / denominator
    if not math.isfinite(normalized):
        raise MitigationMetricError("normalized signed violation is not finite")
    return normalized


def _validated_direction(direction: Any) -> str:
    if direction not in CONSTRAINT_DIRECTIONS:
        raise MitigationMetricError(
            "constraint direction must be 'upper_bound' or 'lower_bound'"
        )
    return str(direction)


def _validated_episode_bounds(
    fault_injection_time: Any,
    episode_horizon: Any,
) -> tuple[float, float]:
    fault_time = _finite_number(fault_injection_time, "fault_injection_time")
    horizon = _finite_number(episode_horizon, "episode_horizon")
    if horizon < fault_time:
        raise MitigationMetricError(
            "episode_horizon must not precede fault_injection_time"
        )
    return fault_time, horizon


def _nonempty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise MitigationMetricError(f"{name} must be a non-empty string")
    return value


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise MitigationMetricError(f"{name} must be a finite number")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise MitigationMetricError(f"{name} must be a finite number")
    return numeric


def _positive_finite_number(value: Any, name: str) -> float:
    numeric = _finite_number(value, name)
    if numeric <= 0.0:
        raise MitigationMetricError(f"{name} must be greater than zero")
    return numeric


def _nonnegative_finite_number(value: Any, name: str) -> float:
    numeric = _finite_number(value, name)
    if numeric < 0.0:
        raise MitigationMetricError(f"{name} must be non-negative")
    return numeric


__all__ = [
    "CONSTRAINT_DIRECTIONS",
    "LOWER_BOUND",
    "MitigationMetricError",
    "SLO_NORMALIZATION_EPSILON",
    "UPPER_BOUND",
    "annotate_stable_recovery",
    "compute_slo_metrics",
    "compute_stable_recovery",
    "make_constraint",
]
