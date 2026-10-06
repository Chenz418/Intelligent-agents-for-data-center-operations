"""Unit tests for manuscript-defined Data Center Twin mitigation metrics."""

from __future__ import annotations

import math

import pytest

from aiopslab.orchestrator.problems.data_center_twin.mitigation_metrics import (
    MitigationMetricError,
    SLO_NORMALIZATION_EPSILON,
    compute_slo_metrics,
    compute_stable_recovery,
    make_constraint,
)


def _constraint(
    observed_value: float,
    threshold: float = 10.0,
    direction: str = "upper_bound",
    *,
    identifier: str = "health_limit",
) -> dict:
    return make_constraint(
        identifier,
        observed_value,
        threshold,
        direction,
        source_field=identifier,
    )


def _sample(
    simulator_time: float,
    healthy: bool,
    constraints: list[dict] | None = None,
) -> dict:
    return {
        "simulator_time": simulator_time,
        "after_fault_injection": simulator_time >= 10.0,
        "constraints": constraints or [],
        "health_condition": healthy,
    }


def _assert_failed_recovery(result: dict, expected_penalty: float) -> None:
    assert result["stable_recovery_succeeded"] is False
    assert result["recovery_start_time"] is None
    assert result["time_to_stable_recovery"] is None
    assert result["penalized_time_to_stable_recovery"] == pytest.approx(
        expected_penalty
    )


def test_make_constraint_preserves_upper_bound_evaluator_inputs():
    constraint = make_constraint(
        "workload_latency_max",
        15.0,
        10.0,
        "upper_bound",
        source_field="workload_average_latency_ms",
        raw_observed_value="15 ms",
    )

    assert constraint["identifier"] == "workload_latency_max"
    assert constraint["source_field"] == "workload_average_latency_ms"
    assert constraint["observed_value"] == 15.0
    assert constraint["raw_observed_value"] == "15 ms"
    assert constraint["threshold"] == 10.0
    assert constraint["direction"] == "upper_bound"
    assert constraint["normalized_signed_violation"] == pytest.approx(0.5)
    assert constraint["healthy"] is False


def test_make_constraint_computes_lower_bound_violation_and_healthy_state():
    violated = make_constraint("capacity_min", 5.0, 10.0, "lower_bound")
    healthy = make_constraint("capacity_min", 12.0, 10.0, "lower_bound")

    assert violated["normalized_signed_violation"] == pytest.approx(0.5)
    assert violated["healthy"] is False
    assert healthy["normalized_signed_violation"] == pytest.approx(-0.2)
    assert healthy["healthy"] is True


def test_make_constraint_uses_epsilon_for_zero_threshold():
    constraint = make_constraint("zero_limit", 2.0, 0.0, "upper_bound")

    assert math.isfinite(constraint["normalized_signed_violation"])
    assert constraint["normalized_signed_violation"] == pytest.approx(
        2.0 / SLO_NORMALIZATION_EPSILON
    )
    assert constraint["healthy"] is False


def test_stable_recovery_is_immediate_when_health_is_sustained_for_full_window():
    trajectory = [
        _sample(10.0, True),
        _sample(12.0, True),
        _sample(15.0, True),
    ]

    result = compute_stable_recovery(
        trajectory,
        fault_injection_time=10.0,
        episode_horizon=30.0,
        stability_window=5.0,
    )

    assert result["stable_recovery_succeeded"] is True
    assert result["recovery_start_time"] == pytest.approx(10.0)
    assert result["time_to_stable_recovery"] == pytest.approx(0.0)
    assert result["penalized_time_to_stable_recovery"] == pytest.approx(0.0)


def test_temporary_recovery_followed_by_relapse_does_not_count():
    trajectory = [
        _sample(10.0, True),
        _sample(12.0, False),
        _sample(15.0, True),
        _sample(20.0, True),
    ]

    result = compute_stable_recovery(
        trajectory,
        fault_injection_time=10.0,
        episode_horizon=30.0,
        stability_window=5.0,
    )

    assert result["stable_recovery_succeeded"] is True
    assert result["recovery_start_time"] == pytest.approx(15.0)
    assert result["time_to_stable_recovery"] == pytest.approx(5.0)
    assert result["penalized_time_to_stable_recovery"] == pytest.approx(5.0)


def test_delayed_sustained_recovery_uses_earliest_qualifying_start():
    trajectory = [
        _sample(10.0, False),
        _sample(13.0, True),
        _sample(16.0, True),
        _sample(18.0, True),
    ]

    result = compute_stable_recovery(
        trajectory,
        fault_injection_time=10.0,
        episode_horizon=30.0,
        stability_window=5.0,
    )

    assert result["stable_recovery_succeeded"] is True
    assert result["recovery_start_time"] == pytest.approx(13.0)
    assert result["time_to_stable_recovery"] == pytest.approx(3.0)
    assert result["penalized_time_to_stable_recovery"] == pytest.approx(3.0)


def test_recovery_beginning_too_late_for_full_window_is_penalized():
    trajectory = [
        _sample(10.0, False),
        _sample(18.0, True),
        _sample(20.0, True),
    ]

    result = compute_stable_recovery(
        trajectory,
        fault_injection_time=10.0,
        episode_horizon=20.0,
        stability_window=5.0,
    )

    _assert_failed_recovery(result, expected_penalty=10.0)


def test_no_recovery_receives_full_remaining_horizon_penalty():
    trajectory = [
        _sample(10.0, False),
        _sample(15.0, False),
        _sample(20.0, False),
    ]

    result = compute_stable_recovery(
        trajectory,
        fault_injection_time=10.0,
        episode_horizon=20.0,
        stability_window=5.0,
    )

    _assert_failed_recovery(result, expected_penalty=10.0)


def test_recovery_endpoint_includes_all_ordered_samples_at_same_timestamp():
    trajectory = [
        _sample(10.0, True),
        _sample(15.0, True),
        _sample(20.0, True),
        _sample(20.0, False),
        _sample(25.0, False),
    ]

    result = compute_stable_recovery(
        trajectory,
        fault_injection_time=10.0,
        episode_horizon=30.0,
        stability_window=10.0,
    )

    _assert_failed_recovery(result, expected_penalty=20.0)


def test_always_healthy_trajectory_has_zero_slo_area_and_duration():
    trajectory = [
        _sample(10.0, True, [_constraint(8.0)]),
        _sample(15.0, True, [_constraint(9.0)]),
    ]

    result = compute_slo_metrics(
        trajectory,
        fault_injection_time=10.0,
        episode_horizon=20.0,
    )

    assert result["slo_violation_area"] == pytest.approx(0.0)
    assert result["raw_slo_violation_duration"] == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("direction", "observed", "expected_area"),
    [
        ("upper_bound", 15.0, 2.0),
        ("lower_bound", 5.0, 2.0),
    ],
)
def test_single_constraint_violation_area(direction, observed, expected_area):
    trajectory = [
        _sample(10.0, False, [_constraint(observed, direction=direction)]),
        _sample(14.0, True, [_constraint(10.0, direction=direction)]),
    ]

    result = compute_slo_metrics(
        trajectory,
        fault_injection_time=10.0,
        episode_horizon=20.0,
    )

    assert result["slo_violation_area"] == pytest.approx(expected_area)
    assert result["raw_slo_violation_duration"] == pytest.approx(4.0)


def test_slo_area_normalizes_by_total_constraint_count():
    trajectory = [
        _sample(
            10.0,
            False,
            [
                _constraint(15.0, identifier="violated"),
                _constraint(8.0, identifier="healthy"),
            ],
        ),
        _sample(
            14.0,
            True,
            [
                _constraint(10.0, identifier="violated"),
                _constraint(8.0, identifier="healthy"),
            ],
        ),
    ]

    result = compute_slo_metrics(
        trajectory,
        fault_injection_time=10.0,
        episode_horizon=20.0,
    )

    # (0.5 + 0.0) * 4 seconds / 2 constraints.
    assert result["slo_violation_area"] == pytest.approx(1.0)
    assert result["raw_slo_violation_duration"] == pytest.approx(4.0)


def test_varying_severity_and_nonuniform_intervals_use_actual_delta_t():
    trajectory = [
        _sample(10.0, False, [_constraint(20.0)]),
        _sample(12.0, False, [_constraint(15.0)]),
        _sample(15.0, True, [_constraint(10.0)]),
    ]

    result = compute_slo_metrics(
        trajectory,
        fault_injection_time=10.0,
        episode_horizon=20.0,
    )

    # 1.0 * 2 seconds + 0.5 * 3 seconds.
    assert result["slo_violation_area"] == pytest.approx(3.5)
    assert result["raw_slo_violation_duration"] == pytest.approx(5.0)


def test_simultaneous_constraint_violations_do_not_double_count_raw_duration():
    trajectory = [
        _sample(
            10.0,
            False,
            [
                _constraint(15.0, identifier="upper"),
                _constraint(5.0, direction="lower_bound", identifier="lower"),
            ],
        ),
        _sample(
            14.0,
            True,
            [
                _constraint(10.0, identifier="upper"),
                _constraint(10.0, direction="lower_bound", identifier="lower"),
            ],
        ),
    ]

    result = compute_slo_metrics(
        trajectory,
        fault_injection_time=10.0,
        episode_horizon=20.0,
    )

    # Two 0.5 violations average to 0.5 for four seconds.
    assert result["slo_violation_area"] == pytest.approx(2.0)
    assert result["raw_slo_violation_duration"] == pytest.approx(4.0)


def test_slo_integration_ignores_prefault_time_and_clips_to_episode_horizon():
    trajectory = [
        _sample(8.0, False, [_constraint(20.0)]),
        _sample(10.0, False, [_constraint(20.0)]),
        _sample(25.0, True, [_constraint(10.0)]),
    ]

    result = compute_slo_metrics(
        trajectory,
        fault_injection_time=10.0,
        episode_horizon=20.0,
    )

    assert result["slo_violation_area"] == pytest.approx(10.0)
    assert result["raw_slo_violation_duration"] == pytest.approx(10.0)


def test_zero_threshold_slo_area_uses_configured_epsilon():
    trajectory = [
        _sample(10.0, False, [_constraint(2.0, threshold=0.0)]),
        _sample(12.0, True, [_constraint(0.0, threshold=0.0)]),
    ]

    result = compute_slo_metrics(
        trajectory,
        fault_injection_time=10.0,
        episode_horizon=20.0,
        epsilon=SLO_NORMALIZATION_EPSILON,
    )

    assert math.isfinite(result["slo_violation_area"])
    assert result["slo_violation_area"] == pytest.approx(
        4.0 / SLO_NORMALIZATION_EPSILON
    )
    assert result["raw_slo_violation_duration"] == pytest.approx(2.0)


def test_slo_rejects_constraint_definition_changes_between_samples():
    trajectory = [
        _sample(10.0, False, [_constraint(15.0, threshold=10.0)]),
        _sample(12.0, True, [_constraint(11.0, threshold=11.0)]),
    ]

    with pytest.raises(MitigationMetricError, match="definition changed"):
        compute_slo_metrics(
            trajectory,
            fault_injection_time=10.0,
            episode_horizon=20.0,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("normalized_signed_violation", 0.0),
        ("healthy", True),
    ],
)
def test_slo_rejects_inconsistent_saved_constraint_derived_fields(field, value):
    corrupted = _constraint(15.0)
    corrupted[field] = value
    trajectory = [
        _sample(10.0, False, [corrupted]),
        _sample(12.0, True, [_constraint(10.0)]),
    ]

    with pytest.raises(MitigationMetricError, match="inconsistent"):
        compute_slo_metrics(
            trajectory,
            fault_injection_time=10.0,
            episode_horizon=20.0,
        )
