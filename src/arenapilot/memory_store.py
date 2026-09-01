from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from .db import initialize_database
from .tracking import arenapilot_home


KNOWLEDGE_SCHEMA_VERSION = 1


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _hash_json(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _next_scoped_name(
    path: Path,
    *,
    table: str,
    competition_id: str,
    prefix: str,
) -> str:
    initialize_database(path)
    with sqlite3.connect(path) as connection:
        rows = connection.execute(
            f"SELECT name FROM {table} WHERE competition_id = ?",
            (competition_id,),
        ).fetchall()
    highest = 0
    for (name,) in rows:
        value = str(name)
        if value.startswith(prefix) and value[len(prefix) :].isdigit():
            highest = max(highest, int(value[len(prefix) :]))
    return f"{prefix}{highest + 1:03d}"


def create_fingerprint_record(
    path: Path,
    *,
    competition_id: str,
    fingerprint: dict[str, object],
    source: str,
) -> dict[str, object]:
    initialize_database(path)
    encoded = _canonical_json(fingerprint)
    digest = _hash_json(fingerprint)
    now = _utc_now()
    with sqlite3.connect(path) as connection:
        connection.row_factory = sqlite3.Row
        existing = connection.execute(
            """
            SELECT * FROM competition_fingerprints
            WHERE competition_id = ? AND fingerprint_hash = ?
            ORDER BY created_at DESC LIMIT 1
            """,
            (competition_id, digest),
        ).fetchone()
        if existing is not None:
            return dict(existing)
        identifier = f"fp_{uuid4().hex}"
        connection.execute(
            """
            INSERT INTO competition_fingerprints(
                id, competition_id, schema_version, fingerprint_json,
                fingerprint_hash, source, created_at
            ) VALUES (?, ?, 1, ?, ?, ?, ?)
            """,
            (identifier, competition_id, encoded, digest, source, now),
        )
        row = connection.execute(
            "SELECT * FROM competition_fingerprints WHERE id = ?",
            (identifier,),
        ).fetchone()
    assert row is not None
    return dict(row)


def latest_fingerprint(path: Path, competition_id: str) -> dict[str, object] | None:
    initialize_database(path)
    with sqlite3.connect(path) as connection:
        connection.row_factory = sqlite3.Row
        row = connection.execute(
            """
            SELECT * FROM competition_fingerprints
            WHERE competition_id = ?
            ORDER BY created_at DESC LIMIT 1
            """,
            (competition_id,),
        ).fetchone()
    return dict(row) if row is not None else None


def create_evidence_record(
    path: Path,
    *,
    competition_id: str,
    subject_type: str,
    subject_key: str,
    outcome: str,
    summary: str,
    strength: int,
    context: dict[str, object],
    validation_domain_hash: str | None = None,
    effect: float | None = None,
    source_experiment_id: str | None = None,
    source_run_id: str | None = None,
    source_submission_id: str | None = None,
    reference_experiment_id: str | None = None,
    reference_run_id: str | None = None,
) -> dict[str, object]:
    initialize_database(path)
    name = _next_scoped_name(
        path,
        table="memory_evidence",
        competition_id=competition_id,
        prefix="evidence",
    )
    identifier = f"ev_{uuid4().hex}"
    with sqlite3.connect(path) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute(
            """
            INSERT INTO memory_evidence(
                id, competition_id, name, subject_type, subject_key,
                source_experiment_id, source_run_id, source_submission_id,
                reference_experiment_id, reference_run_id,
                validation_domain_hash, outcome, effect, strength,
                summary, context_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                identifier,
                competition_id,
                name,
                subject_type,
                subject_key,
                source_experiment_id,
                source_run_id,
                source_submission_id,
                reference_experiment_id,
                reference_run_id,
                validation_domain_hash,
                outcome,
                effect,
                strength,
                summary,
                _canonical_json(context),
                _utc_now(),
            ),
        )
    row = get_evidence(path, competition_id, name)
    assert row is not None
    return row


def get_evidence(path: Path, competition_id: str, name: str) -> dict[str, object] | None:
    initialize_database(path)
    with sqlite3.connect(path) as connection:
        connection.row_factory = sqlite3.Row
        row = connection.execute(
            """
            SELECT * FROM memory_evidence
            WHERE competition_id = ? AND name = ?
            """,
            (competition_id, name),
        ).fetchone()
    return dict(row) if row is not None else None


def list_evidence(path: Path, competition_id: str) -> list[dict[str, object]]:
    initialize_database(path)
    with sqlite3.connect(path) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            """
            SELECT * FROM memory_evidence
            WHERE competition_id = ?
            ORDER BY CAST(SUBSTR(name, 9) AS INTEGER), name
            """,
            (competition_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def create_finding_record(
    path: Path,
    *,
    competition_id: str,
    subject_type: str,
    subject_key: str,
    conclusion: str,
    summary: str,
    confidence: str,
    evidence_links: list[tuple[str, str]],
) -> dict[str, object]:
    initialize_database(path)
    if not evidence_links:
        raise ValueError("KNOWLEDGE_EVIDENCE_MISSING: finding requires evidence")
    name = _next_scoped_name(
        path,
        table="findings",
        competition_id=competition_id,
        prefix="finding",
    )
    identifier = f"finding_{uuid4().hex}"
    now = _utc_now()
    with sqlite3.connect(path) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.row_factory = sqlite3.Row
        resolved: list[tuple[str, str]] = []
        for evidence_name, role in evidence_links:
            row = connection.execute(
                """
                SELECT id, subject_type, subject_key FROM memory_evidence
                WHERE competition_id = ? AND name = ?
                """,
                (competition_id, evidence_name),
            ).fetchone()
            if row is None:
                raise ValueError(f"evidence not found: {evidence_name}")
            if row["subject_type"] != subject_type or row["subject_key"] != subject_key:
                raise ValueError(
                    f"evidence subject mismatch: {evidence_name} is "
                    f"{row['subject_type']}:{row['subject_key']}"
                )
            resolved.append((str(row["id"]), role))

        connection.execute(
            """
            INSERT INTO findings(
                id, competition_id, name, subject_type, subject_key,
                conclusion, summary, confidence, status, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'candidate', ?, ?)
            """,
            (
                identifier,
                competition_id,
                name,
                subject_type,
                subject_key,
                conclusion,
                summary,
                confidence,
                now,
                now,
            ),
        )
        for evidence_id, role in resolved:
            connection.execute(
                """
                INSERT INTO finding_evidence(finding_id, evidence_id, role)
                VALUES (?, ?, ?)
                """,
                (identifier, evidence_id, role),
            )
    row = get_finding(path, competition_id, name)
    assert row is not None
    return row


def get_finding(path: Path, competition_id: str, name: str) -> dict[str, object] | None:
    initialize_database(path)
    with sqlite3.connect(path) as connection:
        connection.row_factory = sqlite3.Row
        row = connection.execute(
            "SELECT * FROM findings WHERE competition_id = ? AND name = ?",
            (competition_id, name),
        ).fetchone()
        if row is None:
            return None
        payload = dict(row)
        links = connection.execute(
            """
            SELECT e.*, fe.role
            FROM finding_evidence fe
            JOIN memory_evidence e ON e.id = fe.evidence_id
            WHERE fe.finding_id = ?
            ORDER BY e.name
            """,
            (row["id"],),
        ).fetchall()
    payload["evidence"] = [dict(item) for item in links]
    return payload


def list_findings(path: Path, competition_id: str) -> list[dict[str, object]]:
    initialize_database(path)
    with sqlite3.connect(path) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            """
            SELECT * FROM findings
            WHERE competition_id = ?
            ORDER BY CAST(SUBSTR(name, 8) AS INTEGER), name
            """,
            (competition_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def approve_finding_record(path: Path, competition_id: str, name: str) -> dict[str, object]:
    initialize_database(path)
    with sqlite3.connect(path) as connection:
        cursor = connection.execute(
            """
            UPDATE findings SET status = 'approved', updated_at = ?
            WHERE competition_id = ? AND name = ? AND status = 'candidate'
            """,
            (_utc_now(), competition_id, name),
        )
        if cursor.rowcount != 1:
            current = connection.execute(
                "SELECT status FROM findings WHERE competition_id = ? AND name = ?",
                (competition_id, name),
            ).fetchone()
            actual = current[0] if current else "missing"
            raise ValueError(f"finding cannot be approved from status {actual}: {name}")
    row = get_finding(path, competition_id, name)
    assert row is not None
    return row


def approved_findings(path: Path, competition_id: str) -> list[dict[str, object]]:
    initialize_database(path)
    with sqlite3.connect(path) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            """
            SELECT * FROM findings
            WHERE competition_id = ? AND status = 'approved'
            ORDER BY created_at
            """,
            (competition_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def knowledge_db_path() -> Path:
    return arenapilot_home() / "knowledge.db"


def initialize_knowledge_database(path: Path | None = None) -> Path:
    target = path or knowledge_db_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(target) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute(
            "CREATE TABLE IF NOT EXISTS schema_meta (version INTEGER NOT NULL)"
        )
        row = connection.execute("SELECT version FROM schema_meta LIMIT 1").fetchone()
        if row is None:
            connection.execute(
                "INSERT INTO schema_meta(version) VALUES (?)",
                (KNOWLEDGE_SCHEMA_VERSION,),
            )
        elif int(row[0]) != KNOWLEDGE_SCHEMA_VERSION:
            raise RuntimeError(f"unsupported knowledge schema version: {row[0]}")

        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS knowledge_items (
                id TEXT PRIMARY KEY,
                kind TEXT NOT NULL,
                key TEXT NOT NULL,
                version INTEGER NOT NULL,
                title TEXT NOT NULL,
                summary TEXT NOT NULL,
                applicability_json TEXT NOT NULL,
                confidence TEXT NOT NULL,
                status TEXT NOT NULL,
                independent_competitions INTEGER NOT NULL,
                positive_count INTEGER NOT NULL,
                neutral_count INTEGER NOT NULL,
                negative_count INTEGER NOT NULL,
                directional_consistency REAL,
                supersedes_id TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (supersedes_id) REFERENCES knowledge_items(id),
                UNIQUE(kind, key, version)
            );
            CREATE INDEX IF NOT EXISTS idx_knowledge_items_lookup
                ON knowledge_items(kind, key, version);

            CREATE TABLE IF NOT EXISTS knowledge_evidence (
                id TEXT PRIMARY KEY,
                knowledge_id TEXT NOT NULL,
                source_competition_id TEXT NOT NULL,
                source_competition_slug TEXT NOT NULL,
                source_finding_id TEXT NOT NULL,
                source_finding_name TEXT NOT NULL,
                conclusion TEXT NOT NULL,
                strength INTEGER NOT NULL,
                fingerprint_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (knowledge_id) REFERENCES knowledge_items(id),
                UNIQUE(knowledge_id, source_competition_id, source_finding_id)
            );
            CREATE INDEX IF NOT EXISTS idx_knowledge_evidence_item
                ON knowledge_evidence(knowledge_id);
            """
        )
    return target


def latest_knowledge_item(
    path: Path,
    kind: str,
    key: str,
) -> dict[str, object] | None:
    initialize_knowledge_database(path)
    with sqlite3.connect(path) as connection:
        connection.row_factory = sqlite3.Row
        row = connection.execute(
            """
            SELECT * FROM knowledge_items
            WHERE kind = ? AND key = ?
            ORDER BY version DESC LIMIT 1
            """,
            (kind, key),
        ).fetchone()
    return dict(row) if row is not None else None


def knowledge_evidence(path: Path, knowledge_id: str) -> list[dict[str, object]]:
    initialize_knowledge_database(path)
    with sqlite3.connect(path) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            """
            SELECT * FROM knowledge_evidence
            WHERE knowledge_id = ?
            ORDER BY source_competition_slug, source_finding_name
            """,
            (knowledge_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def create_knowledge_version(
    path: Path,
    *,
    kind: str,
    key: str,
    title: str,
    summary: str,
    applicability: dict[str, object],
    confidence: str,
    independent_competitions: int,
    positive_count: int,
    neutral_count: int,
    negative_count: int,
    directional_consistency: float | None,
    evidence: list[dict[str, object]],
    supersedes_id: str | None,
) -> dict[str, object]:
    initialize_knowledge_database(path)
    with sqlite3.connect(path) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        row = connection.execute(
            "SELECT MAX(version) FROM knowledge_items WHERE kind = ? AND key = ?",
            (kind, key),
        ).fetchone()
        version = int(row[0] or 0) + 1
        identifier = f"knowledge_{uuid4().hex}"
        connection.execute(
            """
            INSERT INTO knowledge_items(
                id, kind, key, version, title, summary, applicability_json,
                confidence, status, independent_competitions,
                positive_count, neutral_count, negative_count,
                directional_consistency, supersedes_id, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'candidate', ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                identifier,
                kind,
                key,
                version,
                title,
                summary,
                _canonical_json(applicability),
                confidence,
                independent_competitions,
                positive_count,
                neutral_count,
                negative_count,
                directional_consistency,
                supersedes_id,
                _utc_now(),
            ),
        )
        for item in evidence:
            connection.execute(
                """
                INSERT INTO knowledge_evidence(
                    id, knowledge_id, source_competition_id,
                    source_competition_slug, source_finding_id,
                    source_finding_name, conclusion, strength,
                    fingerprint_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    f"ke_{uuid4().hex}",
                    identifier,
                    item["source_competition_id"],
                    item["source_competition_slug"],
                    item["source_finding_id"],
                    item["source_finding_name"],
                    item["conclusion"],
                    int(item["strength"]),
                    _canonical_json(item["fingerprint"]),
                    _utc_now(),
                ),
            )
    result = latest_knowledge_item(path, kind, key)
    assert result is not None
    return result


def list_latest_knowledge(path: Path) -> list[dict[str, object]]:
    initialize_knowledge_database(path)
    with sqlite3.connect(path) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            """
            SELECT k.*
            FROM knowledge_items k
            JOIN (
                SELECT kind, key, MAX(version) AS version
                FROM knowledge_items GROUP BY kind, key
            ) latest
              ON latest.kind = k.kind
             AND latest.key = k.key
             AND latest.version = k.version
            WHERE k.status = 'candidate'
            ORDER BY k.kind, k.key
            """
        ).fetchall()
    return [dict(row) for row in rows]
