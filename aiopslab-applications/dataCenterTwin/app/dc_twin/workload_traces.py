"""CSV and JSONL workload trace replay support."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dc_twin.workload_classes import build_workload_class


class TraceReplayError(ValueError):
    """Raised when a workload trace cannot be loaded or replayed."""


@dataclass(frozen=True)
class TraceReplayRecord:
    timestamp_seconds: int
    tenant_id: str | None
    workload_class: str
    request_rate_per_second: float
    cpu_demand: float | None = None
    memory_demand: float | None = None
    network_demand: float | None = None
    storage_demand: float | None = None

    def resource_overrides(self) -> dict[str, float]:
        return {
            key: value
            for key, value in {
                "cpu": self.cpu_demand,
                "memory": self.memory_demand,
                "network": self.network_demand,
                "storage": self.storage_demand,
            }.items()
            if value is not None
        }


@dataclass(frozen=True)
class TraceReplayProgress:
    enabled: bool
    path: str | None = None
    format: str | None = None
    records_loaded: int = 0
    current_timestamp_seconds: int | None = None
    current_index: int = -1
    tenant_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "path": self.path,
            "format": self.format,
            "records_loaded": self.records_loaded,
            "current_timestamp_seconds": self.current_timestamp_seconds,
            "current_index": self.current_index,
            "tenant_id": self.tenant_id,
        }


class TraceReplayEngine:
    def __init__(self, path: str, records: list[TraceReplayRecord], trace_format: str):
        if not records:
            raise TraceReplayError(f"trace {path} has no valid records")
        self.path = path
        self.trace_format = trace_format
        self.records = sorted(records, key=lambda record: (record.timestamp_seconds, record.tenant_id or ""))
        self._records_by_tenant: dict[str | None, list[TraceReplayRecord]] = {}
        for record in self.records:
            self._records_by_tenant.setdefault(record.tenant_id, []).append(record)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "TraceReplayEngine":
        path = config.get("path") or config.get("trace_path")
        if not path:
            raise TraceReplayError("trace replay requires path")
        trace_format = str(config.get("format") or config.get("trace_format") or Path(path).suffix.lstrip(".")).lower()
        if trace_format not in {"csv", "jsonl"}:
            raise TraceReplayError(f"unsupported trace format: {trace_format}")
        trace_path = Path(path)
        if not trace_path.exists():
            raise TraceReplayError(f"trace file not found: {path}")
        records = _load_csv(trace_path) if trace_format == "csv" else _load_jsonl(trace_path)
        return cls(path=str(trace_path), records=records, trace_format=trace_format)

    def record_at(self, sim_time_seconds: int, tenant_id: str | None = None) -> TraceReplayRecord | None:
        records = self._records_by_tenant.get(tenant_id, [])
        if not records and tenant_id is not None:
            return None
        current: TraceReplayRecord | None = None
        for record in records:
            if record.timestamp_seconds <= sim_time_seconds:
                current = record
            else:
                break
        return current

    def progress(self, sim_time_seconds: int, tenant_id: str | None = None) -> TraceReplayProgress:
        records = self._records_by_tenant.get(tenant_id, [])
        current_index = -1
        current_timestamp = None
        for index, record in enumerate(records):
            if record.timestamp_seconds <= sim_time_seconds:
                current_index = index
                current_timestamp = record.timestamp_seconds
            else:
                break
        return TraceReplayProgress(
            enabled=True,
            path=self.path,
            format=self.trace_format,
            records_loaded=len(records),
            current_timestamp_seconds=current_timestamp,
            current_index=current_index,
            tenant_id=tenant_id,
        )

    def tenant_ids(self) -> list[str]:
        return sorted(tenant_id for tenant_id in self._records_by_tenant if tenant_id is not None)


def _load_csv(path: Path) -> list[TraceReplayRecord]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return [
            _parse_record(row, f"{path}:{line_number}")
            for line_number, row in enumerate(csv.DictReader(file), start=2)
        ]


def _load_jsonl(path: Path) -> list[TraceReplayRecord]:
    records = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as error:
                raise TraceReplayError(f"{path}:{line_number}: invalid JSON: {error.msg}") from error
            if not isinstance(row, dict):
                raise TraceReplayError(f"{path}:{line_number}: trace row must be an object")
            records.append(_parse_record(row, f"{path}:{line_number}"))
    return records


def _parse_record(row: dict[str, Any], source: str) -> TraceReplayRecord:
    timestamp = row.get("timestamp", row.get("time", row.get("tick")))
    if _missing(timestamp):
        raise TraceReplayError(f"{source}: missing timestamp or tick")
    workload_type = row.get("workload_type", row.get("workload_class", row.get("job_type", "web_service")))
    if _missing(workload_type):
        raise TraceReplayError(f"{source}: missing workload_type")
    try:
        workload_class = build_workload_class(str(workload_type)).workload_class
    except ValueError as error:
        raise TraceReplayError(f"{source}: {error}") from error
    request_rate = row.get("request_rate", row.get("request_rate_per_second", row.get("job_demand")))
    if _missing(request_rate):
        raise TraceReplayError(f"{source}: missing request_rate or job_demand")
    timestamp_seconds = int(float(timestamp))
    if timestamp_seconds < 0:
        raise TraceReplayError(f"{source}: timestamp must be >= 0")
    request_rate_per_second = _required_non_negative_float(request_rate, source, "request_rate")
    return TraceReplayRecord(
        timestamp_seconds=timestamp_seconds,
        tenant_id=_optional_str(row.get("tenant_id")),
        workload_class=workload_class,
        request_rate_per_second=request_rate_per_second,
        cpu_demand=_optional_normalized_float(row.get("cpu_demand"), source, "cpu_demand"),
        memory_demand=_optional_normalized_float(row.get("memory_demand"), source, "memory_demand"),
        network_demand=_optional_normalized_float(row.get("network_demand"), source, "network_demand"),
        storage_demand=_optional_normalized_float(row.get("storage_demand"), source, "storage_demand"),
    )


def _required_non_negative_float(value: Any, source: str, field_name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise TraceReplayError(f"{source}: {field_name} must be numeric") from error
    if parsed < 0.0:
        raise TraceReplayError(f"{source}: {field_name} must be >= 0")
    return parsed


def _optional_normalized_float(value: Any, source: str, field_name: str) -> float | None:
    if _missing(value):
        return None
    parsed = _required_non_negative_float(value, source, field_name)
    if parsed > 1.0:
        raise TraceReplayError(f"{source}: {field_name} must be <= 1")
    return parsed


def _optional_str(value: Any) -> str | None:
    if _missing(value):
        return None
    return str(value)


def _missing(value: Any) -> bool:
    return value is None or value == ""
