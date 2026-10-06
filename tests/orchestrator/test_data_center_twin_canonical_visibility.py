from copy import deepcopy
import hashlib
import json

import pytest

from aiopslab.orchestrator.problems.data_center_twin.visibility import (
    AgentVisibilityLeakageError,
    sanitize_canonical_telemetry,
)
from dc_twin.faults import FaultRequest
from dc_twin.simulator import DataCenterSimulator


def _snapshot():
    simulator = DataCenterSimulator()
    simulator.start_workload(
        {"request_rate_per_second": 400, "noise_enabled": False}
    )
    simulator.step(3)
    simulator.inject_fault(
        FaultRequest(
            fault_type="application_error",
            target="application",
            severity=1.0,
            duration_seconds=30,
        )
    )
    simulator.step(3)
    return simulator.canonical_snapshot()


def _rehash(snapshot):
    without_id = deepcopy(snapshot)
    without_id.pop("snapshot_id", None)
    serialized = json.dumps(
        without_id,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    ).encode("utf-8")
    snapshot["snapshot_id"] = (
        "snapshot-" + hashlib.sha256(serialized).hexdigest()[:24]
    )


def test_orchestrator_accepts_an_exact_canonical_snapshot():
    snapshot = _snapshot()

    assert sanitize_canonical_telemetry(snapshot) == snapshot


def test_orchestrator_canonical_boundary_fails_closed_after_digest_recompute():
    snapshot = _snapshot()
    metric_index = next(
        index
        for index, observation in enumerate(snapshot["observations"])
        if observation["channel"] == "metric"
    )
    alert_index = next(
        index
        for index, observation in enumerate(snapshot["observations"])
        if observation["channel"] == "alert"
    )
    log_index = next(
        index
        for index, observation in enumerate(snapshot["observations"])
        if observation["channel"] == "log"
    )

    def top_level_oracle(value):
        value["oracle"] = "hidden"

    def host_visibility(value):
        value["policy"]["host_visibility"] = "host"

    def observation_sidecar(value):
        value["observations"][0]["training_label"] = "positive"

    def payload_extension(value):
        value["observations"][metric_index]["payload"]["hidden_score"] = 1.0

    def nested_oracle(value):
        value["observations"][alert_index]["payload"]["details"][
            "oracle"
        ] = "hidden"

    def invalid_statistic(value):
        value["observations"][metric_index]["payload"]["statistics"][
            "count"
        ] = "1"

    def future_ingest(value):
        value["observations"][0]["metadata"]["ingest_time_seconds"] += 1.0

    def public_sequence_counter(value):
        value["query_watermark_sequence"] = 12
        for observation in value["observations"]:
            observation["metadata"]["available_at_sequence"] = 12

    def missing_provenance(value):
        value["observations"][0]["metadata"]["entities"][0][
            "provenance"
        ] = ""

    def duplicate_observation_id(value):
        value["observations"][1]["observation_id"] = value["observations"][0][
            "observation_id"
        ]

    def exact_host_identifier(value):
        value["observations"][log_index]["payload"]["variable_summaries"][
            "upstream_resource"
        ] = {
            "count": 1,
            "top_values": [
                {
                    "value": "server-r1-row1-rack01-01",
                    "count": 1,
                }
            ],
        }

    for mutate in (
        top_level_oracle,
        host_visibility,
        observation_sidecar,
        payload_extension,
        nested_oracle,
        invalid_statistic,
        future_ingest,
        public_sequence_counter,
        missing_provenance,
        duplicate_observation_id,
        exact_host_identifier,
    ):
        tampered = deepcopy(snapshot)
        mutate(tampered)
        _rehash(tampered)
        with pytest.raises(AgentVisibilityLeakageError):
            sanitize_canonical_telemetry(tampered)
