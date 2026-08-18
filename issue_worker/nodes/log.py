"""Per-node JSONL trace logging for the pipeline.

Pure logging module: no timing, no control flow. Callers compute
duration_ms and pass every field explicitly.
"""

import json
import logging
from pathlib import Path
from datetime import datetime, timezone
import uuid

LOG_PATH = Path("data/pipeline_log.jsonl")

class JSONFormatter(logging.Formatter):
    """Formats each log record's `extra` payload as one JSON line."""

    def format(self, record: logging.LogRecord) ->str:
        payload = record.__dict__.get("event", {})
        return json.dumps(payload, default = str)

def _get_logger() -> logging.Logger:
    logger = logging.getLogger("pipeline_trace")
    if not logger.handlers:
        LOG_PATH.parent.mkdir(parents= True, exist_ok = True)
        handler = logging.FileHandler(LOG_PATH)
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger

def log_event(
    node_name: str,
    source_id: str,
    run_id: str,
    outcome: str,
    failure_reason: str | None = None,
    duration_ms: float | None = None,
    attempt: int | None = None,
    ) -> None:
    """Write one JSONL trace line for a single node call.
    Caller is responsible for timing (duration_ms) — this function
    only formats and writes.
    attempt is optional and only meaningful for nodes with internal
    retry loops (currently only Retry); all other nodes omit it and
    it defaults to None.
    """
    event = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "node_name": node_name,
        "source_id": source_id,
        "run_id": run_id,
        "outcome": outcome,
        "failure_reason": failure_reason,
        "duration_ms": duration_ms,
        "attempt": attempt,
    }
    logger = _get_logger()
    logger.info("", extra={"event": event})



