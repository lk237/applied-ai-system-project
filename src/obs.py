"""
Observability: structured run logs and error capture.

Every request writes one JSON object per line to logs/run-YYYY-MM-DD.jsonl.
JSONL rather than prose because the point of these logs is to be read by a
program later - the eval harness and the README's reliability section both parse
them, and a human debugging a bad recommendation can grep one.

What gets recorded is chosen so a past run can be reconstructed without having
it in front of you: the query, which songs were retrieved and at what
similarity, which guardrails fired, how long each stage took, and the exact
model/config that produced it.
"""

from __future__ import annotations

import json
import platform
import sys
import time
import traceback
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

from src.config import LOG_DIR, describe


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class RunLogger:
    """
    Append-only JSONL logger for one process.

    Failures to write a log are swallowed and reported to stderr rather than
    raised: losing a log line is bad, but crashing a working recommendation
    because the disk is full is worse.
    """

    def __init__(self, log_dir: Optional[Path] = None, stream: bool = False):
        self.log_dir = Path(log_dir) if log_dir else LOG_DIR
        self.run_id = uuid.uuid4().hex[:12]
        self.stream = stream  # echo events to stderr as they happen
        self._path: Optional[Path] = None

    @property
    def path(self) -> Path:
        if self._path is None:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            self._path = self.log_dir / f"run-{day}.jsonl"
        return self._path

    def event(self, event: str, **fields: Any) -> Dict[str, Any]:
        """Write one event. Returns the record so callers can assert on it."""
        record: Dict[str, Any] = {
            "ts": _utc_now(),
            "run_id": self.run_id,
            "event": event,
        }
        record.update(fields)

        try:
            with open(self.path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, default=str) + "\n")
        except OSError as exc:  # disk full, permissions, read-only mount
            print(f"[obs] could not write log: {exc}", file=sys.stderr)

        if self.stream:
            print(f"[obs] {event} {fields}", file=sys.stderr)

        return record

    def session_start(self) -> None:
        self.event(
            "session_start",
            config=describe(),
            python=platform.python_version(),
            platform=platform.platform(),
        )

    def error(self, where: str, exc: BaseException) -> Dict[str, Any]:
        """Record a caught exception with its type, message, and traceback."""
        return self.event(
            "error",
            where=where,
            error_type=type(exc).__name__,
            message=str(exc),
            traceback=traceback.format_exc(limit=6),
        )

    @contextmanager
    def stage(self, name: str, **fields: Any) -> Iterator[Dict[str, Any]]:
        """
        Time a pipeline stage and log it, whether it succeeds or throws.

            with log.stage("retrieval", k=5) as st:
                st["hits"] = len(results)

        Anything the caller puts into the yielded dict is merged into the log
        record, so a stage can report its own results without a second call.
        """
        extra: Dict[str, Any] = {}
        started = time.perf_counter()
        try:
            yield extra
        except BaseException as exc:
            self.event(
                f"{name}_failed",
                ms=round((time.perf_counter() - started) * 1000, 1),
                error_type=type(exc).__name__,
                message=str(exc),
                **fields,
                **extra,
            )
            raise
        else:
            self.event(
                name,
                ms=round((time.perf_counter() - started) * 1000, 1),
                **fields,
                **extra,
            )


def read_events(path: Path) -> list[Dict[str, Any]]:
    """
    Parse a JSONL log back into records, skipping lines that are not valid JSON.

    Tolerant on purpose: a truncated final line from a killed process should not
    make the whole log unreadable.
    """
    if not Path(path).exists():
        return []

    events: list[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return events
