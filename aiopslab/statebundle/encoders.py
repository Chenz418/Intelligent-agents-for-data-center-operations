"""Modality-specific encoders for canonical StateBundle observations.

The paper permits replaceable pretrained text and time-series backbones.  The
default implementation is deliberately self-contained: a trainable hashed
token encoder and a small GRU temporal encoder make tests and CPU smoke runs
reproducible without downloading model weights.  Callers may inject a temporal
backbone with the same ``forward(sequence, observed_mask)`` contract.

Opaque incident/observation identifiers, source references, and correlation
hashes are never embedded.  They remain available to the inference layer for
joining and duplicate suppression only.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Mapping, Sequence
from typing import Any, Protocol, cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from aiopslab.statebundle.config import StateBundleModelConfig
from aiopslab.statebundle.types import CanonicalObservation, TelemetryChannel


_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_.:/-]+")
_SEVERITIES = ("debug", "info", "warning", "critical")


def _stable_token_id(token: str, vocabulary_size: int) -> int:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return 1 + int.from_bytes(digest, "big") % (vocabulary_size - 1)


def _tokens(text: str) -> list[str]:
    values = [match.group(0).lower() for match in _TOKEN_PATTERN.finditer(text)]
    return values or ["<unknown>"]


def _safe_text(value: Any, *, max_items: int = 64) -> str:
    """Flatten payload content without allowing unbounded recursive records."""

    pieces: list[str] = []

    def visit(item: Any, path: str) -> None:
        if len(pieces) >= max_items:
            return
        if isinstance(item, Mapping):
            for key in sorted(item, key=str):
                visit(item[key], f"{path}.{key}" if path else str(key))
        elif isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
            for index, child in enumerate(item[:max_items]):
                visit(child, f"{path}[{index}]")
        elif item is not None:
            pieces.append(f"{path}={item}")

    visit(value, "")
    return " ".join(pieces)


def _finite_number(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return default


def _signed_log1p(value: Any) -> float:
    numeric = _finite_number(value)
    return math.copysign(math.log1p(abs(numeric)), numeric)


def _numeric_leaves(value: Any, *, limit: int = 128) -> list[float]:
    leaves: list[float] = []

    def visit(item: Any) -> None:
        if len(leaves) >= limit:
            return
        if isinstance(item, Mapping):
            for key in sorted(item, key=str):
                visit(item[key])
        elif isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
            for child in item:
                visit(child)
        elif isinstance(item, (bool, int, float)):
            numeric = _finite_number(item, float("nan"))
            if math.isfinite(numeric):
                leaves.append(numeric)

    visit(value)
    return leaves


def _sequence_or_empty(value: Any) -> Sequence[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return value
    return ()


def _summary(value: Any, width: int = 8) -> list[float]:
    leaves = _numeric_leaves(value)
    if not leaves:
        result = [0.0] * 8
    else:
        tensor = torch.tensor(leaves, dtype=torch.float64)
        result = [
            _signed_log1p(len(leaves)),
            _signed_log1p(tensor.mean().item()),
            _signed_log1p(tensor.std(unbiased=False).item()),
            _signed_log1p(tensor.min().item()),
            _signed_log1p(tensor.max().item()),
            _signed_log1p(tensor.abs().mean().item()),
            float((tensor == 0).to(torch.float64).mean().item()),
            float((tensor < 0).to(torch.float64).mean().item()),
        ]
    if width <= len(result):
        return result[:width]
    return [*result, *([0.0] * (width - len(result)))]


class HashedTextEncoder(nn.Module):
    """Trainable, deterministic tokenizer/embedding fallback."""

    def __init__(self, vocabulary_size: int, dimension: int, max_tokens: int):
        super().__init__()
        self.vocabulary_size = vocabulary_size
        self.dimension = dimension
        self.max_tokens = max_tokens
        self.embedding = nn.Embedding(vocabulary_size, dimension, padding_idx=0)

    def forward(self, texts: Sequence[str], *, device: torch.device) -> Tensor:
        if not texts:
            return torch.empty((0, self.dimension), device=device)
        encoded: list[list[int]] = []
        for text in texts:
            ids = [
                _stable_token_id(token, self.vocabulary_size)
                for token in _tokens(text)[: self.max_tokens]
            ]
            encoded.append(ids)
        width = max(len(ids) for ids in encoded)
        token_ids = torch.zeros((len(encoded), width), dtype=torch.long, device=device)
        mask = torch.zeros((len(encoded), width), dtype=torch.bool, device=device)
        for row, ids in enumerate(encoded):
            token_ids[row, : len(ids)] = torch.tensor(
                ids, dtype=torch.long, device=device
            )
            mask[row, : len(ids)] = True
        embeddings = self.embedding(token_ids)
        return (embeddings * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(
            dim=1, keepdim=True
        ).clamp_min(1)


class TemporalBackbone(Protocol):
    output_dimension: int

    def __call__(self, sequence: Tensor, observed_mask: Tensor) -> Tensor: ...


class GRUTemporalBackbone(nn.Module):
    """Small default temporal encoder; replaceable with MOMENT or equivalent."""

    def __init__(self, dimension: int):
        super().__init__()
        self.output_dimension = dimension
        self.gru = nn.GRU(2, dimension, batch_first=True)

    def forward(self, sequence: Tensor, observed_mask: Tensor) -> Tensor:
        inputs = torch.stack((sequence, observed_mask.to(sequence.dtype)), dim=-1)
        _, hidden = self.gru(inputs)
        return hidden[-1]


class _MetadataEncoder(nn.Module):
    _NUMERIC_WIDTH = 12

    def __init__(self, config: StateBundleModelConfig, text: HashedTextEncoder):
        super().__init__()
        self.text = text
        self.numeric = nn.Linear(self._NUMERIC_WIDTH, config.encoder_dimension)
        self.output = nn.Linear(config.encoder_dimension * 2, config.encoder_dimension)

    def forward(
        self, observations: Sequence[CanonicalObservation], device: torch.device
    ) -> Tensor:
        texts: list[str] = []
        numeric: list[list[float]] = []
        for observation in observations:
            metadata = observation.metadata
            roles = sorted(entity.role.value for entity in metadata.entities)
            # Entity identifiers and opaque correlations are intentionally absent.
            texts.append(
                " ".join(
                    (
                        f"subsystem={metadata.primary_subsystem}",
                        *(f"role={role}" for role in roles),
                    )
                )
            )
            confidences = [entity.confidence for entity in metadata.entities]
            quality = metadata.data_quality
            numeric.append(
                [
                    _signed_log1p(
                        metadata.event_end_time_seconds
                        - metadata.event_start_time_seconds
                    ),
                    _signed_log1p(
                        metadata.available_at_time_seconds
                        - metadata.event_end_time_seconds
                    ),
                    quality.parse_confidence,
                    quality.missingness_fraction,
                    _signed_log1p(quality.delay_seconds),
                    _signed_log1p(len(metadata.entities)),
                    max(confidences, default=0.0),
                    sum(confidences) / max(1, len(confidences)),
                    float(metadata.primary_subsystem != "unknown"),
                    _signed_log1p(len(metadata.correlation_ids)),
                    _signed_log1p(len(quality.validation_flags)),
                    sum(quality.availability_mask.values())
                    / max(1, len(quality.availability_mask)),
                ]
            )
        text_embedding = self.text(texts, device=device)
        numeric_tensor = torch.tensor(numeric, dtype=torch.float32, device=device)
        return self.output(
            torch.cat(
                (text_embedding, torch.tanh(self.numeric(numeric_tensor))), dim=-1
            )
        )


class _AdditiveChannelEncoder(nn.Module):
    """Shared implementation of the paper's additive structured encoders."""

    numeric_width: int = 16

    def __init__(self, config: StateBundleModelConfig, text: HashedTextEncoder):
        super().__init__()
        self.config = config
        self.text = text
        self.text_projection = nn.Linear(
            config.encoder_dimension, config.encoder_dimension
        )
        self.numeric_projection = nn.Linear(
            self.numeric_width, config.encoder_dimension
        )
        self.metadata = _MetadataEncoder(config, text)
        self.normalization = nn.LayerNorm(config.encoder_dimension)

    def text_value(self, observation: CanonicalObservation) -> str:
        raise NotImplementedError

    def numeric_value(self, observation: CanonicalObservation) -> list[float]:
        return _summary(observation.payload, self.numeric_width)

    def forward(
        self, observations: Sequence[CanonicalObservation], device: torch.device
    ) -> Tensor:
        texts = [self.text_value(observation) for observation in observations]
        numeric = [self.numeric_value(observation) for observation in observations]
        text_embedding = self.text(texts, device=device)
        numeric_tensor = torch.tensor(numeric, dtype=torch.float32, device=device)
        combined = (
            self.text_projection(text_embedding)
            + self.numeric_projection(numeric_tensor)
            + self.metadata(observations, device)
        )
        return self.normalization(combined)


class LogEncoder(_AdditiveChannelEncoder):
    def text_value(self, observation: CanonicalObservation) -> str:
        payload = observation.payload
        return " ".join(
            str(payload.get(key, "unknown"))
            for key in ("template", "event_type", "severity")
        )

    def numeric_value(self, observation: CanonicalObservation) -> list[float]:
        payload = observation.payload
        histogram = cast(Mapping[str, Any], payload.get("severity_histogram", {}))
        histogram_total = sum(
            max(0.0, _finite_number(value)) for value in histogram.values()
        )
        severity_parts = [
            max(0.0, _finite_number(histogram.get(name))) / max(1.0, histogram_total)
            for name in _SEVERITIES
        ]
        result = [
            _signed_log1p(payload.get("count")),
            _finite_number(payload.get("rarity")),
            _signed_log1p(payload.get("burst_rate_per_minute")),
            *severity_parts,
            *_summary(payload.get("variable_summaries"), 5),
            *_summary(payload.get("time_features"), 4),
        ]
        return result[: self.numeric_width]


class AlertEncoder(_AdditiveChannelEncoder):
    def text_value(self, observation: CanonicalObservation) -> str:
        payload = observation.payload
        return " ".join(
            str(payload.get(key, "unknown"))
            for key in ("message", "alert_type", "severity", "status", "target")
        )


class TraceEncoder(_AdditiveChannelEncoder):
    def text_value(self, observation: CanonicalObservation) -> str:
        payload = observation.payload
        return (
            " ".join(
                str(payload.get(key, "unknown"))
                for key in ("operation", "source", "destination")
            )
            + " "
            + _safe_text(payload.get("status_counts", {}))
        )


class ConfigurationEncoder(_AdditiveChannelEncoder):
    def text_value(self, observation: CanonicalObservation) -> str:
        payload = observation.payload
        visible = {
            key: payload.get(key)
            for key in (
                "path",
                "value",
                "value_type",
                "previous_value",
                "scope",
                "operation",
            )
        }
        return _safe_text(visible)


class MetricEncoder(nn.Module):
    """Robustly normalize and temporally encode metric-series segments."""

    _NUMERIC_WIDTH = 16

    def __init__(
        self,
        config: StateBundleModelConfig,
        text: HashedTextEncoder,
        temporal_backbone: nn.Module | None = None,
    ):
        super().__init__()
        self.config = config
        self.text = text
        self.temporal = temporal_backbone or GRUTemporalBackbone(
            config.encoder_dimension
        )
        temporal_output_dimension = getattr(self.temporal, "output_dimension", None)
        if (
            isinstance(temporal_output_dimension, bool)
            or not isinstance(temporal_output_dimension, int)
            or temporal_output_dimension < 1
        ):
            raise TypeError(
                "metric temporal backbones must expose a positive integer "
                "output_dimension"
            )
        self.temporal_projection = nn.Linear(
            temporal_output_dimension, config.encoder_dimension
        )
        self.text_projection = nn.Linear(
            config.encoder_dimension, config.encoder_dimension
        )
        self.numeric_projection = nn.Linear(
            self._NUMERIC_WIDTH, config.encoder_dimension
        )
        self.metadata = _MetadataEncoder(config, text)
        self.normalization = nn.LayerNorm(config.encoder_dimension)

    def _series(
        self, observation: CanonicalObservation, device: torch.device
    ) -> tuple[Tensor, Tensor]:
        payload = observation.payload
        raw_values = payload.get("values", [])
        raw_mask = payload.get("missingness_mask", [])
        if not isinstance(raw_values, Sequence) or isinstance(raw_values, (str, bytes)):
            raw_values = []
        if not isinstance(raw_mask, Sequence) or isinstance(raw_mask, (str, bytes)):
            raw_mask = []
        length = max(len(raw_values), len(raw_mask), 1)
        values: list[float] = []
        observed: list[bool] = []
        for index in range(length):
            value = raw_values[index] if index < len(raw_values) else None
            missing = bool(raw_mask[index]) if index < len(raw_mask) else value is None
            numeric = _finite_number(value, 0.0)
            values.append(numeric if not missing else 0.0)
            observed.append(not missing and math.isfinite(numeric))

        reference = payload.get("normalization_reference", {})
        if not isinstance(reference, Mapping):
            reference = {}
        median = _finite_number(reference.get("median"), 0.0)
        iqr = abs(_finite_number(reference.get("iqr"), 0.0))
        denominator = iqr + self.config.metric_normalization_epsilon
        normalized = [
            (value - median) / denominator if is_observed else 0.0
            for value, is_observed in zip(values, observed)
        ]
        sequence = torch.tensor(normalized, dtype=torch.float32, device=device).view(
            1, 1, -1
        )
        mask = torch.tensor(observed, dtype=torch.float32, device=device).view(1, 1, -1)
        sequence = F.interpolate(
            sequence,
            size=self.config.metric_sequence_length,
            mode="linear",
            align_corners=False,
        ).view(-1)
        mask = F.interpolate(
            mask, size=self.config.metric_sequence_length, mode="nearest"
        ).view(-1)
        sequence = sequence.clamp(
            -self.config.metric_clip_value, self.config.metric_clip_value
        )
        return sequence, mask

    def forward(
        self, observations: Sequence[CanonicalObservation], device: torch.device
    ) -> Tensor:
        if not observations:
            return torch.empty((0, self.config.encoder_dimension), device=device)
        series_and_masks = [
            self._series(observation, device) for observation in observations
        ]
        series = torch.stack([item[0] for item in series_and_masks])
        masks = torch.stack([item[1] for item in series_and_masks])
        temporal = self.temporal(series, masks)
        texts = [
            " ".join(
                (
                    str(observation.payload.get("metric_name", "unknown")),
                    str(observation.payload.get("unit", "unknown")),
                    str(observation.payload.get("scale", "unknown")),
                    _safe_text(observation.payload.get("resource", {})),
                )
            )
            for observation in observations
        ]
        text_embedding = self.text(texts, device=device)
        numeric = [
            [
                *_summary(observation.payload.get("statistics", {}), 8),
                *_summary(observation.payload.get("normalization_reference", {}), 4),
                _signed_log1p(observation.payload.get("sample_period_seconds")),
                float(
                    sum(
                        bool(value)
                        for value in _sequence_or_empty(
                            observation.payload.get("missingness_mask")
                        )
                    )
                )
                / max(
                    1,
                    len(
                        _sequence_or_empty(observation.payload.get("missingness_mask"))
                    ),
                ),
                _signed_log1p(
                    len(_sequence_or_empty(observation.payload.get("values")))
                ),
                1.0,
            ]
            for observation in observations
        ]
        numeric_tensor = torch.tensor(numeric, dtype=torch.float32, device=device)
        combined = (
            self.temporal_projection(temporal)
            + self.text_projection(text_embedding)
            + self.numeric_projection(numeric_tensor)
            + self.metadata(observations, device)
        )
        return self.normalization(combined)


class ModalityEncoderRegistry(nn.Module):
    """Dispatch five paper-defined encoders and apply channel projections P_c."""

    def __init__(
        self,
        config: StateBundleModelConfig,
        *,
        metric_temporal_backbone: nn.Module | None = None,
    ):
        super().__init__()
        self.config = config
        text = HashedTextEncoder(
            config.hash_vocabulary_size,
            config.encoder_dimension,
            config.max_text_tokens,
        )
        self.text_encoder = text
        self.encoders = nn.ModuleDict(
            {
                TelemetryChannel.LOG.value: LogEncoder(config, text),
                TelemetryChannel.METRIC.value: MetricEncoder(
                    config, text, metric_temporal_backbone
                ),
                TelemetryChannel.ALERT.value: AlertEncoder(config, text),
                TelemetryChannel.TRACE.value: TraceEncoder(config, text),
                TelemetryChannel.CONFIG.value: ConfigurationEncoder(config, text),
            }
        )
        self.projections = nn.ModuleDict(
            {
                channel.value: nn.Linear(
                    config.encoder_dimension, config.shared_dimension
                )
                for channel in TelemetryChannel
            }
        )

    def forward(self, observations: Sequence[CanonicalObservation]) -> Tensor:
        device = next(self.parameters()).device
        if not observations:
            return torch.empty((0, self.config.shared_dimension), device=device)
        result = torch.empty(
            (len(observations), self.config.shared_dimension), device=device
        )
        for channel in TelemetryChannel:
            indices = [
                index
                for index, observation in enumerate(observations)
                if observation.channel is channel
            ]
            if not indices:
                continue
            selected = [observations[index] for index in indices]
            encoded = self.encoders[channel.value](selected, device)
            projected = F.normalize(self.projections[channel.value](encoded), dim=-1)
            result[torch.tensor(indices, device=device)] = projected
        return result


__all__ = [
    "ConfigurationEncoder",
    "GRUTemporalBackbone",
    "HashedTextEncoder",
    "LogEncoder",
    "MetricEncoder",
    "ModalityEncoderRegistry",
    "TemporalBackbone",
]
