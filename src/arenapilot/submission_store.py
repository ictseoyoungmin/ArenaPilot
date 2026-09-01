from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from .db import initialize_database


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def next_submission_name(path: Path, competition_id: str) -> str:
    initialize_database(path)
    with sqlite3.connect(path) as connection:
        rows = connection.execute(
            "SELECT name FROM submissions WHERE competition_id = ?",
            (competition_id,),
        ).fetchall()
    highest = 0
    for (name,) in rows:
        if name.startswith("sub") and name[3:].isdigit():
            highest = max(highest, int(name[3:]))
    return f"sub{highest + 1:03d}"


def create_submission_record(
    path: Path,
    *,
    competition_id: str,
    source_run_id: str,
    file_path: str,
    file_sha256: str,
    platform: str,
    message: str | None = None,
) -> dict[str, object]:
    initialize_database(path)
    name = next_submission_name(path, competition_id)
    submission_id = f"sub_{uuid4().hex}"
    with sqlite3.connect(path) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute(
            """
            INSERT INTO submissions(
                id, competition_id, name, source_run_id, file_path,
                file_sha256, status, platform, message, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, 'created', ?, ?, ?)
            """,
            (
                submission_id,
                competition_id,
                name,
                source_run_id,
                file_path,
                file_sha256,
                platform,
                message,
                _utc_now(),
            ),
        )
    record = get_submission(path, competition_id, name)
    assert record is not None
    return record


def get_submission(
    path: Path,
    competition_id: str,
    name: str,
) -> dict[str, object] | None:
    initialize_database(path)
    with sqlite3.connect(path) as connection:
        connection.row_factory = sqlite3.Row
        row = connection.execute(
            """
            SELECT s.*, r.name AS source_run_name, e.name AS experiment_name
            FROM submissions s
            JOIN runs r ON r.id = s.source_run_id
            JOIN experiments e ON e.id = r.experiment_id
            WHERE s.competition_id = ? AND s.name = ?
            """,
            (competition_id, name),
        ).fetchone()
    return dict(row) if row is not None else None


def list_submissions(path: Path, competition_id: str) -> list[dict[str, object]]:
    initialize_database(path)
    with sqlite3.connect(path) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            """
            SELECT s.*, r.name AS source_run_name, e.name AS experiment_name
            FROM submissions s
            JOIN runs r ON r.id = s.source_run_id
            JOIN experiments e ON e.id = r.experiment_id
            WHERE s.competition_id = ?
            ORDER BY CAST(SUBSTR(s.name, 4) AS INTEGER), s.name
            """,
            (competition_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def mark_validated(path: Path, competition_id: str, name: str) -> dict[str, object]:
    return _transition(
        path,
        competition_id=competition_id,
        name=name,
        from_statuses={"created", "validated"},
        to_status="validated",
        timestamp_column="validated_at",
    )


def mark_submitted(
    path: Path,
    *,
    competition_id: str,
    name: str,
    platform_submission_id: str,
    message: str,
) -> dict[str, object]:
    initialize_database(path)
    now = _utc_now()
    with sqlite3.connect(path) as connection:
        cursor = connection.execute(
            """
            UPDATE submissions
            SET status = 'submitted', platform_submission_id = ?,
                message = ?, submitted_at = ?, failure_message = NULL
            WHERE competition_id = ? AND name = ? AND status = 'validated'
            """,
            (platform_submission_id, message, now, competition_id, name),
        )
        if cursor.rowcount != 1:
            current = connection.execute(
                "SELECT status FROM submissions WHERE competition_id = ? AND name = ?",
                (competition_id, name),
            ).fetchone()
            actual = current[0] if current else "missing"
            raise ValueError(f"invalid submission transition for {name}: {actual} -> submitted")
    record = get_submission(path, competition_id, name)
    assert record is not None
    return record


def update_submission_score(
    path: Path,
    *,
    competition_id: str,
    name: str,
    status: str,
    public_score: float | None,
    private_score: float | None,
    failure_message: str | None = None,
) -> dict[str, object]:
    initialize_database(path)
    scored_at = _utc_now() if status == "scored" else None
    with sqlite3.connect(path) as connection:
        cursor = connection.execute(
            """
            UPDATE submissions
            SET status = ?, public_score = ?, private_score = ?,
                scored_at = COALESCE(?, scored_at), failure_message = ?
            WHERE competition_id = ? AND name = ?
              AND status IN ('submitted', 'pending', 'scored', 'failed')
            """,
            (
                status,
                public_score,
                private_score,
                scored_at,
                failure_message,
                competition_id,
                name,
            ),
        )
        if cursor.rowcount != 1:
            raise ValueError(f"submission cannot be synchronized: {name}")
    record = get_submission(path, competition_id, name)
    assert record is not None
    return record


def submission_budget_usage(path: Path, competition_id: str) -> dict[str, int]:
    initialize_database(path)
    with sqlite3.connect(path) as connection:
        total = connection.execute(
            """
            SELECT COUNT(*) FROM submissions
            WHERE competition_id = ? AND submitted_at IS NOT NULL
            """,
            (competition_id,),
        ).fetchone()[0]
        daily = connection.execute(
            """
            SELECT COUNT(*) FROM submissions
            WHERE competition_id = ? AND submitted_at IS NOT NULL
              AND date(submitted_at) = date('now')
            """,
            (competition_id,),
        ).fetchone()[0]
    return {"daily": int(daily), "total": int(total)}


def _transition(
    path: Path,
    *,
    competition_id: str,
    name: str,
    from_statuses: set[str],
    to_status: str,
    timestamp_column: str | None = None,
) -> dict[str, object]:
    initialize_database(path)
    assignments = ["status = ?"]
    values: list[object] = [to_status]
    if timestamp_column:
        assignments.append(f"{timestamp_column} = ?")
        values.append(_utc_now())
    placeholders = ", ".join("?" for _ in from_statuses)
    values.extend([competition_id, name, *sorted(from_statuses)])
    with sqlite3.connect(path) as connection:
        cursor = connection.execute(
            f"""
            UPDATE submissions
            SET {', '.join(assignments)}
            WHERE competition_id = ? AND name = ?
              AND status IN ({placeholders})
            """,
            values,
        )
        if cursor.rowcount != 1:
            current = connection.execute(
                "SELECT status FROM submissions WHERE competition_id = ? AND name = ?",
                (competition_id, name),
            ).fetchone()
            actual = current[0] if current else "missing"
            raise ValueError(f"invalid submission transition for {name}: {actual} -> {to_status}")
    record = get_submission(path, competition_id, name)
    assert record is not None
    return record
