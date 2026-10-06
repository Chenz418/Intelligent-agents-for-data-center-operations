"""StateBundle observation-boundary tests for the controlled evaluator."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import pytest

from aiopslab.orchestrator.problems.data_center_twin.visibility import (
    assert_no_agent_leakage,
    sanitize_agent_observation,
)
from aiopslab.statebundle.runtime import StateBundleObservationProcessor
from scripts import evaluate_data_center_twin as evaluator


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG = REPO_ROOT / "configs" / "statebundle.dc_twin.yaml"
CHECKPOINT = REPO_ROOT / "checkpoints" / "statebundle-stage3" / "stage3-epoch-9.pt"
PROBLEM = "data_center_twin-cooling_degradation-detection-1"


@pytest.fixture(scope="module")
def processor() -> StateBundleObservationProcessor:
    return StateBundleObservationProcessor(
        config_path=CONFIG,
        checkpoint_path=CHECKPOINT,
    )


def statebundle_args(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        agent="tool-calling",
        model="",
        base_url="",
        api_key="",
        api_key_env="DC_TWIN_LLM_API_KEY",
        provider="",
        output_dir=tmp_path,
        max_steps=3,
        timeout_seconds=100.0,
        temperature=0.0,
        max_tokens=1024,
        reasoning_effort=None,
        rate_limit_max_retries=0,
        rate_limit_initial_delay_seconds=1.0,
        rate_limit_max_delay_seconds=1.0,
        telemetry_view="canonical",
        observation_condition="statebundle",
        observation_token_budget=4096,
        statebundle_config=CONFIG,
        statebundle_checkpoint=CHECKPOINT,
        problem_filter=None,
        expected_problem_count=None,
        task_id_nonce="statebundle-test",
        seed=42,
        allow_auto_advance=False,
        verbose=False,
        debug_logs=False,
        list_problems=False,
        list_agents=False,
        fail_fast=False,
    )


def test_checkpoint_loads_as_exact_stage3_runtime(processor) -> None:
    metadata = processor.public_metadata()

    assert metadata["checkpoint_stage"] == "stage3"
    assert metadata["checkpoint_epoch"] == 9
    assert metadata["checkpoint_global_step"] == 16812
    assert metadata["checkpoint_sha256"] == (
        "9c87538a325bbc52380577ecc4a89ecc7eb989ba2682fb851fc9d5c8c15f29b6"
    )
    assert metadata["input_schema_version"] == "statebundle.canonical.v1"
    assert metadata["output_schema_version"] == "statebundle.output.v1"
    assert metadata["canonical_snapshot_retained"] is False
    assert metadata["model_evaluation_mode"] is True
    assert metadata["parameters_frozen"] is True
    assert processor.model.training is False
    assert all(
        not parameter.requires_grad for parameter in processor.model.parameters()
    )


def test_trained_processor_emits_only_safe_statebundle_output(processor) -> None:
    app = evaluator.InProcessDataCenterTwinApp()
    app.agent_reset(seed=42, stabilization_ticks=1)
    snapshot = app.agent_telemetry(lookback_seconds=300)

    output = processor.transform(snapshot)
    safe = sanitize_agent_observation(output)
    assert_no_agent_leakage(safe)

    assert snapshot["schema_version"] == "statebundle.canonical.v1"
    assert safe["schema_version"] == "statebundle.output.v1"
    assert "observations" not in safe
    assert isinstance(safe["evidence_groups"], list)
    assert processor.audit_records[-1]["input_observation_count"] == len(
        snapshot["observations"]
    )
    assert processor.audit_records[-1]["canonical_snapshot_retained"] is False
    assert processor.audit_records[-1]["preprocessing_latency_seconds"] >= 0.0


def test_statebundle_condition_rejects_missing_checkpoint(tmp_path):
    args = statebundle_args(tmp_path)
    args.statebundle_checkpoint = tmp_path / "missing.pt"
    with pytest.raises(FileNotFoundError):
        evaluator.prepare_observation_condition(args)


def test_controlled_runner_has_no_full_snapshot_side_channel(
    tmp_path, native_submit_agent
) -> None:
    args = statebundle_args(tmp_path)
    evaluator.prepare_observation_condition(args)

    result = asyncio.run(
        evaluator.run_problem(
            PROBLEM,
            args=args,
            output_dir=tmp_path,
            max_steps=3,
            timeout_seconds=100.0,
            seed=42,
            agent_name="tool-calling",
            deterministic=True,
            agent_task_nonce="statebundle-test",
            fail_fast=False,
            debug_logs=True,
            verbose=False,
        )
    )

    assert result["observation_condition"] == "statebundle"
    assert result["telemetry_view"] == "canonical"
    initial = result["initial_agent_visible_observation"]
    assert initial["schema_version"] == "agent.telemetry.compact.v1"
    assert initial["source_schema_version"] == "statebundle.output.v1"
    assert initial["tables"]
    assert initial["budget"]["within_budget"] is True
    agent_visible = {
        "initial": result["initial_agent_visible_observation"],
        "trajectory": result["agent_visible_observations"],
        "actions": result["action_sequence"],
    }
    serialized = json.dumps(agent_visible, sort_keys=True)
    assert "statebundle.canonical.v1" not in serialized
    assert '"observations"' not in serialized
    assert result["observation_pipeline_audit"]
    assert all(
        record["input_schema_version"] == "statebundle.canonical.v1"
        and record["output_schema_version"] == "statebundle.output.v1"
        and record["canonical_snapshot_retained"] is False
        for record in result["observation_pipeline_audit"]
    )
    assert result["max_agent_turns"] == 3
    assert result["timeout_seconds"] == 100.0

    evaluator.write_results(
        tmp_path,
        [result],
        run_config=evaluator.evaluation_run_config(args, [PROBLEM]),
    )
    saved = json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))
    assert saved["results"][0]["observation_condition"] == "statebundle"
    assert saved["run_config"]["statebundle"]["checkpoint_epoch"] == 9
    csv_header = (tmp_path / "results.csv").read_text(encoding="utf-8").splitlines()[0]
    assert "observation_condition" in csv_header
    assert "observation_pipeline_audit" in csv_header


def test_blackbox_statebundle_uses_full_transform_and_compact_workspace_boundary(
    processor,
    tmp_path,
) -> None:
    from clients.data_center_twin_baselines.blackbox_harness import (
        BlackboxEpisodeConfig,
        run_blackbox_problem,
    )

    fake_agent = tmp_path / "fake_statebundle_blackbox.py"
    fake_agent.write_text(
        """
from pathlib import Path
import json
import subprocess
import sys

task_file = Path(sys.argv[1])
workspace = task_file.parent
initial = json.loads((workspace / "INITIAL_OBSERVATION.json").read_text())
assert initial["schema_version"] == "agent.telemetry.compact.v1"
assert initial["source_schema_version"] == "statebundle.output.v1"
assert "observations" not in initial
tool = workspace / "dc_twin_tool.py"
submission = {"incident_detected": False, "evidence": []}
subprocess.run(
    [sys.executable, str(tool), "submit", "--json", json.dumps(submission)],
    cwd=workspace,
    check=True,
)
""",
        encoding="utf-8",
    )
    audit_start = len(processor.audit_records)

    result = run_blackbox_problem(
        PROBLEM,
        BlackboxEpisodeConfig(
            agent_name="codex",
            command_template=f"{sys.executable} {fake_agent} {{task_file}}",
            output_dir=tmp_path,
            max_steps=2,
            timeout_seconds=100.0,
            seed=42,
            sandbox_mode="none",
            telemetry_view="canonical",
            observation_condition="statebundle",
            statebundle_processor=processor,
        ),
    )

    initial = result["initial_agent_visible_observation"]
    assert result["blackbox"]["exit_code"] == 0
    assert result["observation_condition"] == "statebundle"
    assert initial["condition"] == "statebundle"
    assert initial["source_schema_version"] == "statebundle.output.v1"
    assert initial["budget"]["within_budget"] is True
    assert result["statebundle"]["checkpoint_epoch"] == 9
    assert result["observation_pipeline_audit"] == processor.audit_records[audit_start:]
    assert result["observation_pipeline_audit"]
    assert all(
        record["input_schema_version"] == "statebundle.canonical.v1"
        and record["canonical_snapshot_retained"] is False
        for record in result["observation_pipeline_audit"]
    )


pytestmark = pytest.mark.usefixtures("semantic_transport")
