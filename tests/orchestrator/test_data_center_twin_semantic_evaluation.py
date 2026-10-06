"""Judge contract/visibility regressions; semantic ability is tested live separately.

The canned transports below test routing and strict validation, not LLM accuracy.
"""

import json
from types import SimpleNamespace

import pytest

from aiopslab.orchestrator.problems.data_center_twin import cooling_failure as tasks
from aiopslab.orchestrator.problems.data_center_twin import (
    semantic_evaluation as semantic,
)
from scripts import evaluate_data_center_twin as runner


VISIBLE = [
    {
        "schema_version": "agent.telemetry.compact.v1",
        "cooling-unit-1": {"effective_capacity_kw": 0, "rated_capacity_kw": 120},
        "rack_temperature_c": 44,
        "alert": "CoolingCapacityDrop",
        "status": "firing",
    }
]
GROUND_TRUTH = {
    "fault_mechanism": "cooling_degradation",
    "faulty_component": "cooling-unit-1",
    "accepted_target_scopes": ["cooling-unit-1", "CU1"],
    "accepted_operational_domains": ["facility thermal", "cooling"],
}


def output(task="detection", **overrides):
    result = {
        "success": True,
        **dict.fromkeys(semantic.TASK_GATES[task], True),
        "reason": "Submitted diagnosis meets each required criterion.",
    }
    result.update(overrides)
    result["success"] = all(result[k] for k in semantic.TASK_GATES[task])
    if "evidence_supported" in result:
        result["supporting_evidence"] = (
            [
                {
                    "answer_excerpt": "capacity is zero",
                    "observation_index": 0,
                    "observation_excerpt": "effective_capacity_kw: 0",
                    "explanation": "Observed capacity supports the stated cooling failure.",
                }
            ]
            if result["evidence_supported"]
            else []
        )
    return result


def api_response(value):
    return {
        "model": "gpt-5.6-luna",
        "choices": [
            {"finish_reason": "stop", "message": {"content": json.dumps(value)}}
        ],
        "usage": {"total_tokens": 73},
    }


def evaluation_input(task="detection", answer="Cooling has failed: capacity is zero."):
    return semantic.make_input(
        task_type=task,
        final_answer=answer,
        observation_history=VISIBLE,
        ground_truth=GROUND_TRUTH,
    )


@pytest.mark.parametrize(
    "task,answer,gates",
    [
        ("detection", "An incident: cooling_degradation; capacity is zero.", {}),
        (
            "detection",
            "The cooling system has lost all effective cooling capacity; capacity is zero.",
            {},
        ),
        ("detection", "Cooling has failed, not monitoring; capacity is zero.", {}),
        ("detection", "Cooling has failed; observed capacity is zero.", {}),
        (
            "detection",
            "Cooling has failed; the coolant pipe has a measured 50 litre leak.",
            {"evidence_supported": False},
        ),
        (
            "detection",
            "An anomaly exists; the temperature is 44 C.",
            {"mechanism_correct": False},
        ),
        ("localization", ["cooling-unit-1"], {}),
        ("localization", ["CU1"], {}),
        ("localization", {"faulty_components": ["cooling-unit-1"]}, {}),
        (
            "localization",
            "Either the cooling unit or some rack.",
            {"target_correct": False},
        ),
        (
            "localization",
            ["cooling-unit-1", "cooling-unit-2"],
            {"target_correct": False},
        ),
        (
            "analysis",
            {
                "mechanism": "cooling_degradation",
                "target": "CU1",
                "domain": "cooling",
                "evidence": "capacity is zero",
            },
            {},
        ),
        (
            "analysis",
            "Cooling failure at unit 2 in the thermal domain; capacity is zero.",
            {"target_correct": False},
        ),
        (
            "analysis",
            "Cooling failure at CU1 in the network domain; capacity is zero.",
            {"domain_correct": False},
        ),
        (
            "analysis",
            "Cooling failure at CU1 in the thermal domain; a coolant leak was measured.",
            {"evidence_supported": False},
        ),
        (
            "analysis",
            "CU1 cannot remove heat in the facility thermal system: its capacity is zero.",
            {},
        ),
    ],
)
def test_diagnostic_contract_keeps_original_content_and_independent_gates(
    task, answer, gates
):
    seen = []
    expected = output(task, **gates)
    judge = semantic.SemanticEvaluator(
        semantic.SemanticEvaluatorConfig(),
        transport=lambda config, request: (
            seen.append(request) or api_response(expected)
        ),
    )
    payload = evaluation_input(task, answer)
    audit = judge.adjudicate(payload)
    assert audit["output"] == expected
    assert audit["success"] == all(expected[k] for k in semantic.TASK_GATES[task])
    assert json.loads(seen[0]["messages"][1]["content"])["final_answer"] == answer
    assert seen[0]["temperature"] == 0
    assert seen[0]["reasoning_effort"] == "none"
    assert seen[0]["response_format"]["json_schema"]["strict"] is True
    assert audit["token_usage"] == {"total_tokens": 73}


@pytest.mark.parametrize(
    "mutate",
    [
        lambda r: r.update(success=False),
        lambda r: r.update(mechanism_correct="true"),
        lambda r: r.update(success=1),
        lambda r: r.pop("reason"),
        lambda r: r.update(domain_correct=True),
        lambda r: r.update(reason="  "),
        lambda r: r.update(supporting_evidence=[]),
        lambda r: r["supporting_evidence"][0].update(observation_index=99),
        lambda r: r["supporting_evidence"][0].update(observation_index=True),
        lambda r: r["supporting_evidence"][0].update(extra="not allowed"),
    ],
)
def test_strict_schema_rejects_invalid_judge_output(mutate):
    bad = output()
    mutate(bad)
    judge = semantic.SemanticEvaluator(
        semantic.SemanticEvaluatorConfig(), lambda *args: api_response(bad)
    )
    with pytest.raises(semantic.SemanticEvaluatorError) as caught:
        judge.adjudicate(evaluation_input())
    assert caught.value.audit["status"] == "evaluator_infrastructure_error"
    assert caught.value.audit["success"] is None
    assert caught.value.audit["output"] == bad


@pytest.mark.parametrize(
    "text",
    [
        "not json",
        '{"success":true,"success":false}',
        '{"success":NaN}',
        "```json\n{}\n```",
    ],
)
def test_malformed_json_is_detected_and_preserved(text):
    response = api_response({})
    response["choices"][0]["message"]["content"] = text
    judge = semantic.SemanticEvaluator(
        semantic.SemanticEvaluatorConfig(), lambda *args: response
    )
    with pytest.raises(semantic.SemanticEvaluatorError) as caught:
        judge.adjudicate(evaluation_input())
    assert caught.value.audit["response"] == response


@pytest.mark.parametrize(
    "task_class,task_type",
    [
        (tasks.DataCenterTwinCoolingDegradationDetection, "detection"),
        (tasks.DataCenterTwinCoolingDegradationLocalization, "localization"),
        (tasks.DataCenterTwinCoolingDegradationAnalysis, "analysis"),
    ],
)
def test_task_routing_uses_only_semantic_judgment_and_raw_answer(
    task_class, task_type, monkeypatch
):
    task = task_class()
    semantic.configure_problem_evaluator(task, SimpleNamespace())
    submitted = {"payload": "Submitted prose unchanged"}
    runner.record_agent_rendered_observation_for_evaluation(
        task, VISIBLE[0], renderer=object()
    )
    task.agent_action_history = [
        {"action_name": "dc_twin_observe", "response": {"hidden": "do not credit"}}
    ]
    seen = []
    task.semantic_evaluator = semantic.SemanticEvaluator(
        semantic.SemanticEvaluatorConfig(),
        lambda config, request: seen.append(request) or api_response(output(task_type)),
    )
    result = task.eval(submitted, [], 2)
    recorded = result["semantic_adjudication"]["input"]
    assert recorded["final_answer"] == submitted
    assert recorded["agent_visible_observation_history"] == VISIBLE
    assert "hidden" not in json.dumps(recorded)
    assert result["success"] is True
    assert len(seen) == 1


def test_api_failure_is_unscored_without_retry_or_agent_token_charge(monkeypatch):
    task = tasks.DataCenterTwinCoolingDegradationDetection()
    semantic.configure_problem_evaluator(task, SimpleNamespace())
    attempts = []

    def fail(config, request):
        attempts.append(config.model)
        raise TimeoutError("provider unavailable")

    task.semantic_evaluator = semantic.SemanticEvaluator(
        semantic.SemanticEvaluatorConfig(), fail
    )
    result = task.eval("cooling failure", [], 1)
    assert result["success"] is None
    assert result["evaluation_status"] == "evaluator_infrastructure_error"
    assert runner.extract_score_accuracy(result, None) == (None, None)
    assert (
        runner.termination_reason_from_state(
            final_state="submitted",
            submitted=True,
            timed_out=False,
            runtime_error=False,
            evaluation_results=result,
        )
        == "benchmark_error"
    )
    assert attempts == ["gpt-5.6-luna"]
    stats = runner.aggregate_results(
        [
            {"success": True},
            {"success": False},
            {"success": None, "evaluator_results": result},
        ]
    )
    assert stats["scored_episode_count"] == 2
    assert stats["success_rate"] == 0.5
    assert stats["evaluator_infrastructure_error_count"] == 1


@pytest.mark.parametrize(
    "task_class",
    [
        tasks.DataCenterTwinCoolingDegradationMitigation,
        tasks.DataCenterTwinStorageIoSaturationMitigation,
        tasks.DataCenterTwinNetworkPartitionMitigation,
    ],
)
@pytest.mark.parametrize("unhealthy_tick", [None, 4])
def test_mitigation_keeps_ten_tick_simulator_check_never_calls_judge(
    task_class, unhealthy_tick, monkeypatch
):
    task = task_class()
    semantic.configure_problem_evaluator(task, SimpleNamespace())
    assert not hasattr(task, "semantic_evaluator")
    ticks = []

    def advance():
        ticks.append(len(ticks))
        return {}, {}, {"healthy": ticks[-1] != unhealthy_tick}

    monkeypatch.setattr(task, "_advance_evaluator_tick", advance)
    monkeypatch.setattr(task, "_capture_evaluator_health_sample", lambda *a: None)
    monkeypatch.setattr(
        task, "_summary_satisfies_success_criteria", lambda s: s["healthy"]
    )
    monkeypatch.setattr(task, "finalize_mitigation_metrics", lambda: {})
    monkeypatch.setattr(
        semantic.SemanticEvaluator,
        "adjudicate",
        lambda *a: pytest.fail("mitigation called LLM"),
    )
    assert task.eval(None, [], 1)["success"] == (unhealthy_tick is None)
    assert len(ticks) == 10


def test_privileged_inputs_are_minimal_and_task_specific():
    for task_type in semantic.TASK_GATES:
        original = {
            **GROUND_TRUTH,
            "mitigation_actions": ["secret"],
            "active_faults": ["secret"],
        }
        result = semantic.make_input(
            task_type=task_type,
            final_answer="answer",
            observation_history=VISIBLE,
            ground_truth=original,
        )
        assert "secret" not in json.dumps(result)
        if task_type == "localization":
            assert "fault_mechanism" not in result["ground_truth"]
        if task_type == "detection":
            assert "accepted_operational_domains" not in result["ground_truth"]
    with pytest.raises(ValueError, match="only"):
        semantic.make_input(
            task_type="mitigation",
            final_answer=None,
            observation_history=[],
            ground_truth={},
        )
