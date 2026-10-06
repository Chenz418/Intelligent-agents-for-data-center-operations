"""Structured JSON logging helpers."""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "level": record.levelname,
            "component": "dc-twin",
            "episode_id": getattr(record, "episode_id", None),
            "event_type": getattr(record, "event_type", "log"),
            "sim_time_seconds": getattr(record, "sim_time_seconds", None),
            "message": record.getMessage(),
            "details": getattr(record, "details", {}),
        }
        return json.dumps(payload, sort_keys=True)


def configure_logging() -> logging.Logger:
    logger = logging.getLogger("dc-twin")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def log_event(
    logger: logging.Logger,
    event_type: str,
    sim_time_seconds: int,
    message: str,
    details: dict[str, Any] | None = None,
    level: int = logging.INFO,
    episode_id: str | None = None,
) -> None:
    logger.log(
        level,
        message,
        extra={
            "episode_id": episode_id,
            "event_type": event_type,
            "sim_time_seconds": sim_time_seconds,
            "details": details or {},
        },
    )

