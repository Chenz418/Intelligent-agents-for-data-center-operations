"""Trusted live-inference adapter for StateBundle benchmark observations.

The adapter deliberately exposes only :class:`StateBundleOutput` dictionaries.
Canonical input snapshots are parsed and consumed in memory, never retained in
the agent trajectory or returned alongside the selected evidence bundle.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import math
import re
from pathlib import Path
import time
from typing import Any, Mapping

import torch

from aiopslab.agent_telemetry import AgentObservationRequest
from aiopslab.statebundle.checkpoints import load_model_checkpoint
from aiopslab.statebundle.config import load_statebundle_config
from aiopslab.statebundle.data import parse_canonical_snapshot
from aiopslab.statebundle.inference import (
    CompactAgentTokenEstimator,
    StateBundleInference,
)
from aiopslab.statebundle.model import StateBundleModel, TrainingStage
from aiopslab.statebundle.types import (
    CANONICAL_SCHEMA_VERSION,
    STATEBUNDLE_OUTPUT_SCHEMA_VERSION,
)


_SAFE_REQUEST_FIELDS = (
    "include_config",
    "channels",
    "lookback_seconds",
    "log_limit",
    "detail",
    "metric_names",
    "entity_ids",
    "subsystem_ids",
    "alert_names",
)
_FORBIDDEN_ACTION_PARAMETER_KEYS = frozenset(
    {
        "accepted",
        "action_schema_ref",
        "action_type",
        "active_faults",
        "active_faults_after",
        "active_faults_before",
        "action_result",
        "alert_names",
        "available_actions",
        "benchmark_action_coverage",
        "channels",
        "detail",
        "entity_ids",
        "error",
        "episode_id",
        "evaluator_state",
        "expected",
        "expected_diagnosis",
        "expected_mitigation",
        "fault",
        "fault_id",
        "fault_mechanism",
        "fault_target",
        "fault_type",
        "ground_truth",
        "host_visibility",
        "http_status",
        "include_action_schema",
        "include_config",
        "incident_domains",
        "injected_faults",
        "log_limit",
        "lookback_seconds",
        "metric_names",
        "observation",
        "oracle",
        "remaining_duration_seconds",
        "root_cause",
        "scenario",
        "score_hints",
        "solution",
        "sim_time_seconds",
        "sim_time_seconds_after",
        "sim_time_seconds_before",
        "step_summary",
        "subsystem_ids",
        "success_criteria",
        "training_labels",
    }
)


def _normalize_observation_request(
    request: AgentObservationRequest | Mapping[str, Any] | None,
) -> AgentObservationRequest:
    """Rebuild one request using only the public, validated query controls."""

    supported = set(_SAFE_REQUEST_FIELDS)
    if request is None:
        return AgentObservationRequest()
    if isinstance(request, Mapping):
        unexpected = sorted(str(key) for key in request if key not in supported)
        if unexpected:
            raise ValueError(
                f"unsupported StateBundle observation request field(s): {unexpected}"
            )
        values = {name: request[name] for name in supported if name in request}
    elif isinstance(request, AgentObservationRequest):
        values = {name: getattr(request, name) for name in supported}
    else:
        raise TypeError(
            "StateBundle observation request must be AgentObservationRequest, "
            "an object, or None"
        )
    return AgentObservationRequest(**values)


def _sanitize_action_parameter(value: Any, *, path: str) -> Any:
    """Copy an agent-issued action value while rejecting non-JSON state."""

    if value is None or isinstance(value, (bool, str, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"StateBundle action context is non-finite at {path}")
        return value
    if isinstance(value, Mapping):
        sanitized: dict[str, Any] = {}
        for raw_key, child in sorted(value.items(), key=lambda item: str(item[0])):
            if not isinstance(raw_key, str) or not raw_key.strip():
                raise TypeError(
                    f"StateBundle action parameter keys must be text at {path}"
                )
            key = raw_key.strip()
            normalized_key = re.sub(r"[^a-z0-9]+", "_", key.lower()).strip("_")
            if normalized_key in _FORBIDDEN_ACTION_PARAMETER_KEYS:
                continue
            sanitized[key] = _sanitize_action_parameter(
                child,
                path=f"{path}.{key}",
            )
        return sanitized
    if isinstance(value, (list, tuple)):
        return [
            _sanitize_action_parameter(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    raise TypeError(
        f"StateBundle action context contains unsupported {type(value).__name__} "
        f"at {path}"
    )


def _normalize_action_context(
    action: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Allow only an agent-issued action type and its control parameters."""

    if action is None:
        return None
    if not isinstance(action, Mapping):
        raise TypeError("StateBundle action context must be an object or None")
    action_type = action.get("action_type")
    if not isinstance(action_type, str) or not action_type.strip():
        raise ValueError("StateBundle action context requires a non-empty action_type")
    parameters = action.get("parameters", {})
    if parameters is None:
        parameters = {}
    if not isinstance(parameters, Mapping):
        raise TypeError("StateBundle action parameters must be an object")
    return {
        "action_type": action_type.strip(),
        "parameters": _sanitize_action_parameter(parameters, path="action.parameters"),
    }


class StateBundleObservationProcessor:
    """Load one trained selector and transform canonical snapshots in process."""

    def __init__(
        self,
        *,
        config_path: str | Path,
        checkpoint_path: str | Path,
        token_budget: int | None = None,
    ) -> None:
        self.config_path = Path(config_path).expanduser().resolve(strict=True)
        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve(strict=True)
        self.config = load_statebundle_config(self.config_path)
        self.model = StateBundleModel(
            self.config.model,
            self.config.root_cause_catalog,
        )
        self.checkpoint_info = load_model_checkpoint(
            self.model,
            self.checkpoint_path,
            runtime_config=self.config,
            expected_stage=TrainingStage.EVIDENCE_SELECTION,
        )
        if token_budget is not None:
            if (
                isinstance(token_budget, bool)
                or not isinstance(token_budget, int)
                or token_budget < 1
            ):
                raise ValueError("token_budget must be a positive integer")
            self.config = replace(
                self.config,
                inference=replace(
                    self.config.inference,
                    anchor_token_budget=min(
                        self.config.inference.anchor_token_budget,
                        token_budget,
                    ),
                    total_token_budget=token_budget,
                ),
            )
        self.token_budget_override = token_budget
        self.device = torch.device(self.config.device)
        self.model.to(self.device)
        # Evaluation runners must never leave the frozen production selector in
        # a trainable state between calls.  StateBundleInference also guards the
        # forward pass, but making the runtime contract explicit here prevents
        # accidental adaptation by any caller holding the processor.
        self.model.eval()
        self.model.requires_grad_(False)
        self.inference = StateBundleInference(
            self.model,
            self.config.inference,
            token_estimator=CompactAgentTokenEstimator(),
        )
        self.audit_records: list[dict[str, Any]] = []

    @staticmethod
    def _payload_sha256(value: Mapping[str, Any]) -> str:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def begin_episode(self) -> None:
        self.inference.begin_episode()

    def activate_episode(self) -> None:
        self.inference.activate_episode()

    def end_episode(self) -> None:
        self.inference.end_episode()

    def transform(
        self,
        snapshot: Mapping[str, Any],
        *,
        request: AgentObservationRequest | Mapping[str, Any] | None = None,
        action: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return only selected, redacted StateBundle evidence for one snapshot."""

        if not isinstance(snapshot, Mapping):
            raise TypeError("StateBundle observation input must be an object")
        if snapshot.get("schema_version") != CANONICAL_SCHEMA_VERSION:
            raise ValueError(
                f"StateBundle observation input must use {CANONICAL_SCHEMA_VERSION!r}"
            )
        started_at = time.perf_counter()
        incident = parse_canonical_snapshot(snapshot)
        safe_request = _normalize_observation_request(request)
        safe_action = _normalize_action_context(action)
        inference_result = self.inference.infer(
            incident, request=safe_request, action=safe_action
        )
        result = inference_result.to_dict()
        audit_dict = getattr(inference_result, "audit_dict", None)
        selection_diagnostics = audit_dict() if callable(audit_dict) else {}
        preprocessing_latency_seconds = time.perf_counter() - started_at
        if result.get("schema_version") != STATEBUNDLE_OUTPUT_SCHEMA_VERSION:
            raise RuntimeError("StateBundle inference returned an unexpected schema")

        selected_count = sum(
            1 + len(group.get("corroborating_observations", []))
            for group in result.get("evidence_groups", [])
            if isinstance(group, dict)
        )
        self.audit_records.append(
            {
                "call_index": len(self.audit_records) + 1,
                "input_schema_version": snapshot.get("schema_version"),
                "input_snapshot_id": snapshot.get("snapshot_id"),
                "input_observation_count": len(incident.observations),
                "input_snapshot_sha256": self._payload_sha256(snapshot),
                "output_schema_version": result.get("schema_version"),
                "output_snapshot_id": result.get("snapshot_id"),
                "output_selected_observation_count": selected_count,
                "output_sha256": self._payload_sha256(result),
                "canonical_snapshot_retained": False,
                "preprocessing_latency_seconds": preprocessing_latency_seconds,
                # Trusted audit-only diagnostics.  Candidate observations are
                # represented by IDs and observable policy features; this
                # record is never returned to the agent.
                "selection_diagnostics": selection_diagnostics,
            }
        )
        return deepcopy(result)

    def public_metadata(self) -> dict[str, Any]:
        """Return reproducibility metadata with no model tensors or observations."""

        return {
            "processor": type(self).__name__,
            "config_path": str(self.config_path),
            "config_sha256": hashlib.sha256(self.config_path.read_bytes()).hexdigest(),
            "checkpoint_path": str(self.checkpoint_path),
            "checkpoint_sha256": self.checkpoint_info.sha256,
            "checkpoint_schema_version": self.checkpoint_info.schema_version,
            "checkpoint_stage": self.checkpoint_info.stage.value,
            "checkpoint_epoch": self.checkpoint_info.epoch,
            "checkpoint_global_step": self.checkpoint_info.global_step,
            "device": str(self.device),
            "model_evaluation_mode": not self.model.training,
            "parameters_frozen": all(
                not parameter.requires_grad for parameter in self.model.parameters()
            ),
            "parameter_tensor_count": sum(1 for _ in self.model.parameters()),
            "input_schema_version": CANONICAL_SCHEMA_VERSION,
            "output_schema_version": STATEBUNDLE_OUTPUT_SCHEMA_VERSION,
            "canonical_snapshot_retained": False,
            "effective_agent_token_budget": self.config.inference.total_token_budget,
            "token_budget_override": self.token_budget_override,
        }


__all__ = ["StateBundleObservationProcessor"]
