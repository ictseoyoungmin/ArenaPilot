from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from .db import initialize_database


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def create_remote_job(
    path: Path,
    *,
    run_id: str,
    provider: str,
    provider_job_id: str,
    bundle_path: str,
) -> dict[str, object]:
    initialize_database(path)
    job_id = f"rjob_{uuid4().hex}"
    now = _utc_now()
    with sqlite3.connect(path) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.row_factory = sqlite3.Row
        run = connection.execute(
            "SELECT id, remote_job_id FROM runs WHERE id = ?",
            (run_id,),
        ).fetchone()
        if run is None:
            raise ValueError(f"run not found for remote job: {run_id}")
        if run["remote_job_id"] is not None:
            raise ValueError(f"run already has a remote job: {run_id}")

        connection.execute(
            """
            INSERT INTO remote_jobs(
                id, run_id, provider, provider_job_id, state,
                bundle_path, submitted_at, last_seen_at, recovery_state
            ) VALUES (?, ?, ?, ?, 'created', ?, NULL, ?, 'pending')
            """,
            (job_id, run_id, provider, provider_job_id, bundle_path, now),
        )
        connection.execute(
            "UPDATE runs SET remote_job_id = ? WHERE id = ?",
            (job_id, run_id),
        )

    record = get_remote_job(path, job_id)
    assert record is not None
    return record


def get_remote_job(path: Path, job_id: str) -> dict[str, object] | None:
    initialize_database(path)
    with sqlite3.connect(path) as connection:
        connection.row_factory = sqlite3.Row
        row = connection.execute(
            "SELECT * FROM remote_jobs WHERE id = ?",
            (job_id,),
        ).fetchone()
    return dict(row) if row is not None else None


def get_remote_job_for_run(path: Path, run_id: str) -> dict[str, object] | None:
    initialize_database(path)
    with sqlite3.connect(path) as connection:
        connection.row_factory = sqlite3.Row
        row = connection.execute(
            "SELECT * FROM remote_jobs WHERE run_id = ?",
            (run_id,),
        ).fetchone()
    return dict(row) if row is not None else None


def update_remote_job(
    path: Path,
    job_id: str,
    *,
    state: str | None = None,
    recovery_state: str | None = None,
) -> dict[str, object]:
    initialize_database(path)
    assignments = ["last_seen_at = ?"]
    values: list[object] = [_utc_now()]

    if state is not None:
        assignments.append("state = ?")
        values.append(state)
        if state == "submitted":
            assignments.append("submitted_at = COALESCE(submitted_at, ?)")
            values.append(_utc_now())
        elif state == "running":
            assignments.append("started_at = COALESCE(started_at, ?)")
            values.append(_utc_now())
        elif state in {"completed", "failed"}:
            assignments.append("finished_at = COALESCE(finished_at, ?)")
            values.append(_utc_now())

    if recovery_state is not None:
        assignments.append("recovery_state = ?")
        values.append(recovery_state)

    values.append(job_id)
    with sqlite3.connect(path) as connection:
        cursor = connection.execute(
            f"UPDATE remote_jobs SET {', '.join(assignments)} WHERE id = ?",
            values,
        )
        if cursor.rowcount != 1:
            raise ValueError(f"remote job not found: {job_id}")

    record = get_remote_job(path, job_id)
    assert record is not None
    return record
