"""Append-only metric logs that stay correct across crashes and resumes.

metrics.jsonl / val_metrics.jsonl are append-only: rows are never rewritten
(history is evidence).  A crash after logging steps beyond the last
COMMITTED checkpoint, followed by a resume from that checkpoint, therefore
leaves duplicate rows for the replayed steps (Codex round 4: 1,2,3,4,3,4).

Each training session (attempt) appends one line to sessions.jsonl with the
step it started from, and stamps its metric rows with its attempt number.
read_metrics() returns the RECONCILED history: a row is superseded when a
later attempt restarted from an earlier step (so it replayed that step on
the committed state).  Raw files keep every row from every attempt.

Rows written before sessions existed carry no attempt number.  They are
treated as attempt 0, so an old run with no sessions.jsonl reads unchanged.
"""

from __future__ import annotations

import json
import pathlib
import time

SESSIONS = "sessions.jsonl"


def _read_jsonl(path: pathlib.Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def read_sessions(exp_dir: str | pathlib.Path) -> list[dict]:
    """All recorded training sessions, oldest first."""
    return _read_jsonl(pathlib.Path(exp_dir) / SESSIONS)


def start_session(exp_dir: str | pathlib.Path, start_step: int) -> int:
    """Record a new training session starting at `start_step` (the step of
    the committed checkpoint it resumed from, 0 for a fresh run).  Returns
    its attempt number (1, 2, ...)."""
    exp_dir = pathlib.Path(exp_dir)
    attempt = len(read_sessions(exp_dir)) + 1
    exp_dir.mkdir(parents=True, exist_ok=True)
    with open(exp_dir / SESSIONS, "a", encoding="utf-8") as f:
        f.write(json.dumps({"attempt": attempt, "start_step": start_step,
                            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S")}) + "\n")
    return attempt


def read_metrics(exp_dir: str | pathlib.Path, name: str = "metrics.jsonl") -> list[dict]:
    """Reconciled rows of a metric log, in file order.  A row from attempt a
    at step s is dropped if some later attempt a' > a started from a step
    < s: that attempt resumed from committed state before s and replayed s."""
    exp_dir = pathlib.Path(exp_dir)
    starts = {s["attempt"]: s["start_step"] for s in read_sessions(exp_dir)}
    rows = _read_jsonl(exp_dir / name)
    kept = []
    for row in rows:
        a = row.get("attempt", 0)
        if any(later > a and start < row["step"] for later, start in starts.items()):
            continue
        kept.append(row)
    return kept
