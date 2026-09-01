from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def next_run_name(path: Path, competition_id: str) -> str:
    with sqlite3.connect(path) as connection:
        rows = connection.execute(
            "SELECT name FROM runs WHERE competition_id = ?",
            (competition_id,),
        ).fetchall()
    highest = 0
    for (name,) in rows:
        if name.startswith("run") and name[3:].isdigit():
            highest = max(highest, int(name[3:]))
    return f"run{highest + 1:03d}"


def create_run_record(
    path: Path,
    *,
    competition_id: str,
    experiment_id: str,
    backend: str,
    spec_hash: str,
    retry_of: str | None = None,
) -> dict[str, object]:
    name = next_run_name(path, competition_id)
    run_id = f"run_{uuid4().hex}"
    retry_id: str | None = None

    with sqlite3.connect(path) as connection:
        connection.row_factory = sqlite3.Row
        if retry_of is not None:
            retry_row = connection.execute(
                "SELECT id, experiment_id FROM runs WHERE competition_id = ? AND name = ?",
                (competition_id, retry_of),
            ).fetchone()
            if retry_row is None:
                raise ValueError(f"retry run not found: {retry_of}")
            if retry_row["experiment_id"] != experiment_id:
                raise ValueError("retry run belongs to a different experiment")
            retry_id = str(retry_row["id"])

        connection.execute(
            """
            INSERT INTO runs(
                id, competition_id, experiment_id, name, status,
                backend, retry_of, spec_hash, created_at
            ) VALUES (?, ?, ?, ?, 'created', ?, ?, ?, ?)
            """,
            (
                run_id,
                competition_id,
                experiment_id,
                name,
                backend,
                retry_id,
                spec_hash,
                _utc_now(),
            ),
        )

    record = get_run(path, competition_id, name)
    assert record is not None
    return record


def get_run(path: Path, competition_id: str, name: str) -> dict[str, object] | None:
    with sqlite3.connect(path) as connection:
        connection.row_factory = sqlite3.Row
        row = connection.execute(
            """
            SELECT r.*, e.name AS experiment_name, e.title AS experiment_title
            FROM runs r
            JOIN experiments e ON e.id = r.experiment_id
            WHERE r.competition_id = ? AND r.name = ?
            """,
            (competition_id, name),
        ).fetchone()
    return dict(row) if row is not None else None


def list_runs(path: Path, competition_id: str) -> list[dict[str, object]]:
    with sqlite3.connect(path) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            """
            SELECT r.*, e.name AS experiment_name, e.title AS experiment_title
            FROM runs r
            JOIN experiments e ON e.id = r.experiment_id
            WHERE r.competition_id = ?
            ORDER BY CAST(SUBSTR(r.name, 4) AS INTEGER), r.name
            """,
            (competition_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def get_canonical_run_for_experiment(
    path: Path,
    competition_id: str,
    experiment_name: str,
) -> dict[str, object] | None:
    with sqlite3.connect(path) as connection:
        connection.row_factory = sqlite3.Row
        row = connection.execute(
            """
            SELECT r.*, e.name AS experiment_name, e.title AS experiment_title
            FROM experiments e
            JOIN runs r ON r.id = e.canonical_run_id
            WHERE e.competition_id = ? AND e.name = ?
            """,
            (competition_id, experiment_name),
        ).fetchone()
    return dict(row) if row is not None else None


def list_artifact_refs(path: Path, run_id: str) -> list[dict[str, object]]:
    with sqlite3.connect(path) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            """
            SELECT * FROM artifact_refs
            WHERE run_id = ?
            ORDER BY kind, uri
            """,
            (run_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def finalize_verified_run(
    path: Path,
    *,
    competition_id: str,
    name: str,
    manifest_path: str,
    manifest_hash: str,
    mlflow_run_id: str,
    artifacts: list[dict[str, object]],
) -> dict[str, object]:
    now = _utc_now()
    with sqlite3.connect(path) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.row_factory = sqlite3.Row
        row = connection.execute(
            """
            SELECT id, experiment_id, status
            FROM runs
            WHERE competition_id = ? AND name = ?
            """,
            (competition_id, name),
        ).fetchone()
        if row is None:
            raise ValueError(f"run not found: {name}")
        if row["status"] != "verifying":
            raise ValueError(
                f"invalid run transition for {name}: {row['status']} -> verified"
            )

        run_id = str(row["id"])
        connection.execute(
            """
            UPDATE runs
            SET status = 'verified', artifact_manifest_path = ?,
                artifact_manifest_hash = ?, mlflow_run_id = ?, verified_at = ?
            WHERE id = ?
            """,
            (manifest_path, manifest_hash, mlflow_run_id, now, run_id),
        )

        connection.execute("DELETE FROM artifact_refs WHERE run_id = ?", (run_id,))
        for artifact in artifacts:
            connection.execute(
                """
                INSERT INTO artifact_refs(
                    id, run_id, kind, uri, sha256, size_bytes, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    f"art_{uuid4().hex}",
                    run_id,
                    str(artifact["kind"]),
                    str(artifact["uri"]),
                    str(artifact["sha256"]),
                    int(artifact["size_bytes"]),
                    now,
                ),
            )

        connection.execute(
            """
            UPDATE experiments
            SET canonical_run_id = COALESCE(canonical_run_id, ?)
            WHERE id = ?
            """,
            (run_id, str(row["experiment_id"])),
        )

    record = get_run(path, competition_id, name)
    assert record is not None
    return record


def transition_run(
    path: Path,
    *,
    competition_id: str,
    name: str,
    from_statuses: set[str],
    to_status: str,
    exit_code: int | None = None,
    manifest_path: str | None = None,
    manifest_hash: str | None = None,
) -> dict[str, object]:
    timestamp_column: str | None = {
        "queued": "queued_at",
        "running": "started_at",
        "completed": "finished_at",
        "failed": "finished_at",
        "invalid": "finished_at",
        "verified": "verified_at",
    }.get(to_status)

    assignments = ["status = ?"]
    values: list[object] = [to_status]
    if timestamp_column is not None:
        assignments.append(f"{timestamp_column} = ?")
        values.append(_utc_now())
    if exit_code is not None:
        assignments.append("exit_code = ?")
        values.append(exit_code)
    if manifest_path is not None:
        assignments.append("artifact_manifest_path = ?")
        values.append(manifest_path)
    if manifest_hash is not None:
        assignments.append("artifact_manifest_hash = ?")
        values.append(manifest_hash)

    placeholders = ", ".join("?" for _ in from_statuses)
    values.extend([competition_id, name, *sorted(from_statuses)])
    sql = f"""
        UPDATE runs
        SET {', '.join(assignments)}
        WHERE competition_id = ? AND name = ?
          AND status IN ({placeholders})
    """

    with sqlite3.connect(path) as connection:
        cursor = connection.execute(sql, values)
        if cursor.rowcount != 1:
            current = connection.execute(
                "SELECT status FROM runs WHERE competition_id = ? AND name = ?",
                (competition_id, name),
            ).fetchone()
            actual = current[0] if current else "missing"
            raise ValueError(
                f"invalid run transition for {name}: {actual} -> {to_status}"
            )

    record = get_run(path, competition_id, name)
    assert record is not None
    return record
