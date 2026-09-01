from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from .memory_schema import initialize_workspace_memory_schema
from .memory_store import initialize_knowledge_database, knowledge_db_path


PROMOTION_SCHEMA_VERSION = 1


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def initialize_workspace_promotion_schema(path: Path) -> None:
    initialize_workspace_memory_schema(path)
    with sqlite3.connect(path) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute(
            "CREATE TABLE IF NOT EXISTS promotion_schema_meta (version INTEGER NOT NULL)"
        )
        row = connection.execute("SELECT version FROM promotion_schema_meta LIMIT 1").fetchone()
        if row is None:
            connection.execute(
                "INSERT INTO promotion_schema_meta(version) VALUES (?)",
                (PROMOTION_SCHEMA_VERSION,),
            )
        elif int(row[0]) != PROMOTION_SCHEMA_VERSION:
            raise RuntimeError(f"unsupported workspace promotion schema version: {row[0]}")
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS competition_independence_profile (
                competition_id TEXT PRIMARY KEY,
                independence_key TEXT NOT NULL,
                dataset_key TEXT,
                relation TEXT NOT NULL,
                parent_competition_slug TEXT,
                source TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (competition_id) REFERENCES competitions(id)
            );
            """
        )


def initialize_promotion_database(path: Path | None = None) -> Path:
    target = initialize_knowledge_database(path or knowledge_db_path())
    with sqlite3.connect(target) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute(
            "CREATE TABLE IF NOT EXISTS promotion_schema_meta (version INTEGER NOT NULL)"
        )
        row = connection.execute("SELECT version FROM promotion_schema_meta LIMIT 1").fetchone()
        if row is None:
            connection.execute(
                "INSERT INTO promotion_schema_meta(version) VALUES (?)",
                (PROMOTION_SCHEMA_VERSION,),
            )
        elif int(row[0]) != PROMOTION_SCHEMA_VERSION:
            raise RuntimeError(f"unsupported promotion schema version: {row[0]}")
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS competition_independence_registry (
                competition_id TEXT PRIMARY KEY,
                competition_slug TEXT NOT NULL,
                independence_key TEXT NOT NULL,
                dataset_key TEXT,
                relation TEXT NOT NULL,
                parent_competition_slug TEXT,
                fingerprint_hash TEXT,
                source TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_independence_registry_key
                ON competition_independence_registry(independence_key);

            CREATE TABLE IF NOT EXISTS technique_registry (
                key TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                category TEXT NOT NULL,
                description TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_technique_registry_status
                ON technique_registry(status, category, key);

            CREATE TABLE IF NOT EXISTS knowledge_assessments (
                knowledge_id TEXT PRIMARY KEY,
                raw_competitions INTEGER NOT NULL,
                independent_units INTEGER NOT NULL,
                directional_units INTEGER NOT NULL,
                positive_units INTEGER NOT NULL,
                neutral_units INTEGER NOT NULL,
                negative_units INTEGER NOT NULL,
                directional_consistency REAL,
                major_contradiction INTEGER NOT NULL,
                effective_confidence TEXT NOT NULL,
                assessed_at TEXT NOT NULL,
                FOREIGN KEY (knowledge_id) REFERENCES knowledge_items(id)
            );

            CREATE TABLE IF NOT EXISTS knowledge_promotions (
                knowledge_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                approved_confidence TEXT,
                reason TEXT,
                approved_at TEXT,
                deprecated_at TEXT,
                superseded_by_id TEXT,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (knowledge_id) REFERENCES knowledge_items(id),
                FOREIGN KEY (superseded_by_id) REFERENCES knowledge_items(id)
            );
            CREATE INDEX IF NOT EXISTS idx_knowledge_promotions_status
                ON knowledge_promotions(status);
            """
        )
    return target


def set_independence_profile(
    workspace_path: Path,
    global_path: Path,
    *,
    competition_id: str,
    competition_slug: str,
    independence_key: str,
    dataset_key: str | None,
    relation: str,
    parent_competition_slug: str | None,
    fingerprint_hash: str | None,
    source: str,
) -> dict[str, object]:
    initialize_workspace_promotion_schema(workspace_path)
    initialize_promotion_database(global_path)
    now = _utc_now()
    with sqlite3.connect(workspace_path) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute(
            """
            INSERT INTO competition_independence_profile(
                competition_id, independence_key, dataset_key, relation,
                parent_competition_slug, source, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(competition_id) DO UPDATE SET
                independence_key = excluded.independence_key,
                dataset_key = excluded.dataset_key,
                relation = excluded.relation,
                parent_competition_slug = excluded.parent_competition_slug,
                source = excluded.source,
                updated_at = excluded.updated_at
            """,
            (
                competition_id,
                independence_key,
                dataset_key,
                relation,
                parent_competition_slug,
                source,
                now,
            ),
        )
    upsert_global_independence_profile(
        global_path,
        competition_id=competition_id,
        competition_slug=competition_slug,
        independence_key=independence_key,
        dataset_key=dataset_key,
        relation=relation,
        parent_competition_slug=parent_competition_slug,
        fingerprint_hash=fingerprint_hash,
        source=source,
    )
    row = get_workspace_independence_profile(workspace_path, competition_id)
    assert row is not None
    return row


def upsert_global_independence_profile(
    path: Path,
    *,
    competition_id: str,
    competition_slug: str,
    independence_key: str,
    dataset_key: str | None = None,
    relation: str = "independent",
    parent_competition_slug: str | None = None,
    fingerprint_hash: str | None = None,
    source: str = "manual",
) -> None:
    initialize_promotion_database(path)
    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            INSERT INTO competition_independence_registry(
                competition_id, competition_slug, independence_key, dataset_key,
                relation, parent_competition_slug, fingerprint_hash, source, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(competition_id) DO UPDATE SET
                competition_slug = excluded.competition_slug,
                independence_key = excluded.independence_key,
                dataset_key = excluded.dataset_key,
                relation = excluded.relation,
                parent_competition_slug = excluded.parent_competition_slug,
                fingerprint_hash = excluded.fingerprint_hash,
                source = excluded.source,
                updated_at = excluded.updated_at
            """,
            (
                competition_id,
                competition_slug,
                independence_key,
                dataset_key,
                relation,
                parent_competition_slug,
                fingerprint_hash,
                source,
                _utc_now(),
            ),
        )


def get_workspace_independence_profile(path: Path, competition_id: str) -> dict[str, object] | None:
    initialize_workspace_promotion_schema(path)
    with sqlite3.connect(path) as connection:
        connection.row_factory = sqlite3.Row
        row = connection.execute(
            "SELECT * FROM competition_independence_profile WHERE competition_id = ?",
            (competition_id,),
        ).fetchone()
    return dict(row) if row is not None else None


def independence_registry(path: Path) -> dict[str, dict[str, object]]:
    initialize_promotion_database(path)
    with sqlite3.connect(path) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute("SELECT * FROM competition_independence_registry").fetchall()
    return {str(row["competition_id"]): dict(row) for row in rows}


def register_technique_record(
    path: Path,
    *,
    key: str,
    title: str,
    category: str,
    description: str,
) -> dict[str, object]:
    initialize_promotion_database(path)
    now = _utc_now()
    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            INSERT INTO technique_registry(
                key, title, category, description, status, created_at, updated_at
            ) VALUES (?, ?, ?, ?, 'active', ?, ?)
            ON CONFLICT(key) DO UPDATE SET
                title = excluded.title,
                category = excluded.category,
                description = excluded.description,
                status = 'active',
                updated_at = excluded.updated_at
            """,
            (key, title, category, description, now, now),
        )
    row = get_technique_record(path, key)
    assert row is not None
    return row


def get_technique_record(path: Path, key: str) -> dict[str, object] | None:
    initialize_promotion_database(path)
    with sqlite3.connect(path) as connection:
        connection.row_factory = sqlite3.Row
        row = connection.execute("SELECT * FROM technique_registry WHERE key = ?", (key,)).fetchone()
    return dict(row) if row is not None else None


def list_technique_records(path: Path) -> list[dict[str, object]]:
    initialize_promotion_database(path)
    with sqlite3.connect(path) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            "SELECT * FROM technique_registry ORDER BY category, key"
        ).fetchall()
    return [dict(row) for row in rows]


def deprecate_technique_record(path: Path, key: str, reason: str) -> dict[str, object]:
    initialize_promotion_database(path)
    with sqlite3.connect(path) as connection:
        cursor = connection.execute(
            """
            UPDATE technique_registry
            SET status = 'deprecated', description = description || ?, updated_at = ?
            WHERE key = ? AND status = 'active'
            """,
            (f"\nDeprecated: {reason}", _utc_now(), key),
        )
        if cursor.rowcount != 1:
            raise ValueError(f"technique cannot be deprecated: {key}")
    row = get_technique_record(path, key)
    assert row is not None
    return row


def knowledge_version(path: Path, kind: str, key: str, version: int | None = None) -> dict[str, object] | None:
    initialize_promotion_database(path)
    with sqlite3.connect(path) as connection:
        connection.row_factory = sqlite3.Row
        if version is None:
            row = connection.execute(
                """
                SELECT * FROM knowledge_items
                WHERE kind = ? AND key = ? ORDER BY version DESC LIMIT 1
                """,
                (kind, key),
            ).fetchone()
        else:
            row = connection.execute(
                "SELECT * FROM knowledge_items WHERE kind = ? AND key = ? AND version = ?",
                (kind, key, version),
            ).fetchone()
    return dict(row) if row is not None else None


def knowledge_evidence_rows(path: Path, knowledge_id: str) -> list[dict[str, object]]:
    initialize_promotion_database(path)
    with sqlite3.connect(path) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            "SELECT * FROM knowledge_evidence WHERE knowledge_id = ? ORDER BY source_competition_slug, source_finding_name",
            (knowledge_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def save_assessment(path: Path, knowledge_id: str, assessment: dict[str, object]) -> dict[str, object]:
    initialize_promotion_database(path)
    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            INSERT INTO knowledge_assessments(
                knowledge_id, raw_competitions, independent_units, directional_units,
                positive_units, neutral_units, negative_units, directional_consistency,
                major_contradiction, effective_confidence, assessed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(knowledge_id) DO UPDATE SET
                raw_competitions = excluded.raw_competitions,
                independent_units = excluded.independent_units,
                directional_units = excluded.directional_units,
                positive_units = excluded.positive_units,
                neutral_units = excluded.neutral_units,
                negative_units = excluded.negative_units,
                directional_consistency = excluded.directional_consistency,
                major_contradiction = excluded.major_contradiction,
                effective_confidence = excluded.effective_confidence,
                assessed_at = excluded.assessed_at
            """,
            (
                knowledge_id,
                int(assessment["raw_competitions"]),
                int(assessment["independent_units"]),
                int(assessment["directional_units"]),
                int(assessment["positive_units"]),
                int(assessment["neutral_units"]),
                int(assessment["negative_units"]),
                assessment["directional_consistency"],
                1 if assessment["major_contradiction"] else 0,
                str(assessment["effective_confidence"]),
                _utc_now(),
            ),
        )
    return get_assessment(path, knowledge_id) or {}


def get_assessment(path: Path, knowledge_id: str) -> dict[str, object] | None:
    initialize_promotion_database(path)
    with sqlite3.connect(path) as connection:
        connection.row_factory = sqlite3.Row
        row = connection.execute(
            "SELECT * FROM knowledge_assessments WHERE knowledge_id = ?",
            (knowledge_id,),
        ).fetchone()
    return dict(row) if row is not None else None


def get_promotion(path: Path, knowledge_id: str) -> dict[str, object] | None:
    initialize_promotion_database(path)
    with sqlite3.connect(path) as connection:
        connection.row_factory = sqlite3.Row
        row = connection.execute(
            "SELECT * FROM knowledge_promotions WHERE knowledge_id = ?",
            (knowledge_id,),
        ).fetchone()
    return dict(row) if row is not None else None


def approve_knowledge_record(
    path: Path,
    *,
    knowledge: dict[str, object],
    confidence: str,
    reason: str | None,
) -> dict[str, object]:
    initialize_promotion_database(path)
    now = _utc_now()
    knowledge_id = str(knowledge["id"])
    with sqlite3.connect(path) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute(
            """
            INSERT INTO knowledge_promotions(
                knowledge_id, status, approved_confidence, reason,
                approved_at, deprecated_at, superseded_by_id, updated_at
            ) VALUES (?, 'approved', ?, ?, ?, NULL, NULL, ?)
            ON CONFLICT(knowledge_id) DO UPDATE SET
                status = 'approved',
                approved_confidence = excluded.approved_confidence,
                reason = excluded.reason,
                approved_at = COALESCE(knowledge_promotions.approved_at, excluded.approved_at),
                deprecated_at = NULL,
                superseded_by_id = NULL,
                updated_at = excluded.updated_at
            """,
            (knowledge_id, confidence, reason, now, now),
        )
        older = connection.execute(
            """
            SELECT k.id FROM knowledge_items k
            JOIN knowledge_promotions p ON p.knowledge_id = k.id
            WHERE k.kind = ? AND k.key = ? AND k.version < ? AND p.status = 'approved'
            """,
            (knowledge["kind"], knowledge["key"], int(knowledge["version"])),
        ).fetchall()
        for (older_id,) in older:
            connection.execute(
                """
                UPDATE knowledge_promotions
                SET status = 'superseded', superseded_by_id = ?, updated_at = ?
                WHERE knowledge_id = ?
                """,
                (knowledge_id, now, older_id),
            )
    row = get_promotion(path, knowledge_id)
    assert row is not None
    return row


def deprecate_knowledge_record(path: Path, knowledge_id: str, reason: str) -> dict[str, object]:
    initialize_promotion_database(path)
    now = _utc_now()
    with sqlite3.connect(path) as connection:
        existing = connection.execute(
            "SELECT status FROM knowledge_promotions WHERE knowledge_id = ?",
            (knowledge_id,),
        ).fetchone()
        if existing is None:
            connection.execute(
                """
                INSERT INTO knowledge_promotions(
                    knowledge_id, status, approved_confidence, reason,
                    approved_at, deprecated_at, superseded_by_id, updated_at
                ) VALUES (?, 'deprecated', NULL, ?, NULL, ?, NULL, ?)
                """,
                (knowledge_id, reason, now, now),
            )
        elif existing[0] in {"candidate", "approved", "superseded"}:
            connection.execute(
                """
                UPDATE knowledge_promotions
                SET status = 'deprecated', reason = ?, deprecated_at = ?, updated_at = ?
                WHERE knowledge_id = ?
                """,
                (reason, now, now, knowledge_id),
            )
        else:
            raise ValueError(f"knowledge already deprecated: {knowledge_id}")
    row = get_promotion(path, knowledge_id)
    assert row is not None
    return row


def knowledge_history_rows(path: Path, kind: str, key: str) -> list[dict[str, object]]:
    initialize_promotion_database(path)
    with sqlite3.connect(path) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            """
            SELECT k.*, p.status AS promotion_status,
                   p.approved_confidence, p.reason,
                   p.approved_at, p.deprecated_at, p.superseded_by_id,
                   a.independent_units, a.directional_units,
                   a.directional_consistency, a.major_contradiction,
                   a.effective_confidence
            FROM knowledge_items k
            LEFT JOIN knowledge_promotions p ON p.knowledge_id = k.id
            LEFT JOIN knowledge_assessments a ON a.knowledge_id = k.id
            WHERE k.kind = ? AND k.key = ?
            ORDER BY k.version
            """,
            (kind, key),
        ).fetchall()
    return [dict(row) for row in rows]


def retrieval_rows(path: Path) -> list[dict[str, object]]:
    initialize_promotion_database(path)
    with sqlite3.connect(path) as connection:
        connection.row_factory = sqlite3.Row
        subjects = connection.execute(
            "SELECT DISTINCT kind, key FROM knowledge_items ORDER BY kind, key"
        ).fetchall()
        result: list[dict[str, object]] = []
        for subject in subjects:
            approved = connection.execute(
                """
                SELECT k.* FROM knowledge_items k
                JOIN knowledge_promotions p ON p.knowledge_id = k.id
                WHERE k.kind = ? AND k.key = ? AND p.status = 'approved'
                ORDER BY k.version DESC LIMIT 1
                """,
                (subject["kind"], subject["key"]),
            ).fetchone()
            if approved is not None:
                result.append(dict(approved))
                continue
            latest = connection.execute(
                """
                SELECT k.* FROM knowledge_items k
                LEFT JOIN knowledge_promotions p ON p.knowledge_id = k.id
                WHERE k.kind = ? AND k.key = ?
                  AND COALESCE(p.status, 'candidate') != 'deprecated'
                ORDER BY k.version DESC LIMIT 1
                """,
                (subject["kind"], subject["key"]),
            ).fetchone()
            if latest is not None:
                result.append(dict(latest))
    return result
