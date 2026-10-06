"""Strict, CPU-first loading of the bundled StateBundle inference checkpoint."""

from dataclasses import dataclass
from pathlib import Path
import hashlib
import io
from collections.abc import Mapping
import torch
from .model import TrainingStage

CHECKPOINT_SCHEMA_VERSION = "statebundle.checkpoint.v2"
MODEL_SCHEMA_VERSION = "statebundle.model.membership-only/v2"


@dataclass(frozen=True, slots=True)
class CheckpointInfo:
    path: Path
    sha256: str
    schema_version: str
    stage: TrainingStage
    epoch: int
    global_step: int

    def to_dict(self):
        return {
            "path": str(self.path),
            "sha256": self.sha256,
            "schema_version": self.schema_version,
            "stage": self.stage.value,
            "epoch": self.epoch,
            "global_step": self.global_step,
        }


def checkpoint_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_model_checkpoint(
    model, path, *, runtime_config, expected_stage=TrainingStage.EVIDENCE_SELECTION
):
    source = Path(path).expanduser().resolve(strict=True)
    serialized = source.read_bytes()
    payload = torch.load(io.BytesIO(serialized), map_location="cpu", weights_only=True)
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION
    ):
        raise ValueError("unsupported StateBundle checkpoint schema")
    if payload.get("model_schema_version") != MODEL_SCHEMA_VERSION:
        raise ValueError("incompatible StateBundle model schema")
    stage = TrainingStage(payload.get("stage"))
    if stage != expected_stage:
        raise ValueError(f"checkpoint stage must be {expected_stage.value}")
    saved = payload.get("config")
    current = runtime_config.to_dict()
    if not isinstance(saved, Mapping) or any(
        saved.get(key) != current[key] for key in ("root_cause_catalog", "model")
    ):
        raise ValueError("checkpoint model configuration is incompatible")
    state = payload.get("model_state")
    expected = model.state_dict()
    if not isinstance(state, Mapping) or set(state) != set(expected):
        raise ValueError("checkpoint model state has missing or unexpected tensors")
    for key, tensor in state.items():
        reference = expected[key]
        if (
            not isinstance(tensor, torch.Tensor)
            or tensor.shape != reference.shape
            or tensor.dtype != reference.dtype
        ):
            raise ValueError(f"checkpoint tensor mismatch: {key}")
        if (tensor.is_floating_point() or tensor.is_complex()) and not bool(
            torch.isfinite(tensor).all()
        ):
            raise ValueError(f"checkpoint tensor contains NaN or Inf: {key}")
    for key in ("epoch", "global_step"):
        value = payload.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"checkpoint {key} must be a nonnegative integer")
    model.load_state_dict(state, strict=True)
    return CheckpointInfo(
        source,
        hashlib.sha256(serialized).hexdigest(),
        CHECKPOINT_SCHEMA_VERSION,
        stage,
        payload["epoch"],
        payload["global_step"],
    )
