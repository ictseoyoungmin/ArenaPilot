from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from .models import ArenaConfig, ValidationSpec


SCHEMA_VERSION = 1

_SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS schema_meta (
    version INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS competitions (
    id TEXT PRIMARY KEY,
    platform TEXT NOT NULL,
    slug TEXT NOT NULL,
    title TEXT,
    task_type TEXT NOT NULL,
    target TEXT NOT NULL,
    metric_name TEXT NOT NULL,
    metric_direction TEXT NOT NULL,
    active_validation_id TEXT,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(platform, slug)
);

CREATE TABLE IF NOT EXISTS validation_versions (
    id TEXT PRIMARY KEY,
    competition_id TEXT NOT NULL,
    name TEXT NOT NULL,
    parent_id TEXT,
    status TEXT NOT NULL,
    comparison_domain_hash TEXT NOT NULL,
    spec_hash TEXT NOT NULL,
    spec_path TEXT NOT NULL,
    reason TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (competition_id) REFERENCES competitions(id),
    FOREIGN KEY (parent_id) REFERENCES validation_versions(id),
    UNIQUE(competition_id, name)
);

CREATE TABLE IF NOT EXISTS experiments (
    id TEXT PRIMARY KEY,
    competition_id TEXT NOT NULL,
    name TEXT NOT NULL,
    title TEXT,
    status TEXT NOT NULL,
    hypothesis TEXT NOT NULL,
    validation_id TEXT NOT NULL,
    comparison_domain_hash TEXT NOT NULL,
    config_hash TEXT,
    spec_path TEXT NOT NULL,
    evaluation TEXT NOT NULL DEFAULT 'unknown',
    evaluation_note TEXT,
    canonical_run_id TEXT,
    created_at TEXT NOT NULL,
    frozen_at TEXT,
    evaluated_at TEXT,
    FOREIGN KEY (competition_id) REFERENCES competitions(id),
    FOREIGN KEY (validation_id) REFERENCES validation_versions(id),
    UNIQUE(competition_id, name)
);

CREATE TABLE IF NOT EXISTS experiment_parents (
    experiment_id TEXT NOT NULL,
    parent_experiment_id TEXT NOT NULL,
    relation TEXT NOT NULL,
    PRIMARY KEY (experiment_id, parent_experiment_id, relation),
    FOREIGN KEY (experiment_id) REFERENCES experiments(id),
    FOREIGN KEY (parent_experiment_id) REFERENCES experiments(id)
);

CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    competition_id TEXT NOT NULL,
    experiment_id TEXT NOT NULL,
    name TEXT NOT NULL,
    status TEXT NOT NULL,
    backend TEXT NOT NULL,
    retry_of TEXT,
    spec_hash TEXT NOT NULL,
    artifact_manifest_path TEXT,
    artifact_manifest_hash TEXT,
    mlflow_run_id TEXT,
    remote_job_id TEXT,
    exit_code INTEGER,
    created_at TEXT NOT NULL,
    queued_at TEXT,
    started_at TEXT,
    finished_at TEXT,
    verified_at TEXT,
    FOREIGN KEY (competition_id) REFERENCES competitions(id),
    FOREIGN KEY (experiment_id) REFERENCES experiments(id),
    FOREIGN KEY (retry_of) REFERENCES runs(id),
    UNIQUE(competition_id, name)
);
"""


def initialize_database(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as connection:
        connection.executescript(_SCHEMA)
        row = connection.execute("SELECT version FROM schema_meta LIMIT 1").fetchone()
        if row is None:
            connection.execute("INSERT INTO schema_meta(version) VALUES (?)", (SCHEMA_VERSION,))
        elif row[0] != SCHEMA_VERSION:
            raise RuntimeError(f"unsupported database schema version: {row[0]}")


def read_schema_version(path: Path) -> int:
    with sqlite3.connect(path) as connection:
        row = connection.execute("SELECT version FROM schema_meta LIMIT 1").fetchone()
    if row is None:
        raise RuntimeError("database schema version is missing")
    return int(row[0])


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _upsert_competition(
    connection: sqlite3.Connection,
    competition_id: str,
    config: ArenaConfig,
) -> None:
    if config.task is None or config.metric is None:
        raise ValueError("competition intake is incomplete")

    now = _utc_now()
    connection.execute(
        """
        INSERT INTO competitions(
            id, platform, slug, title, task_type, target,
            metric_name, metric_direction, active_validation_id,
            status, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            title = excluded.title,
            task_type = excluded.task_type,
            target = excluded.target,
            metric_name = excluded.metric_name,
            metric_direction = excluded.metric_direction,
            active_validation_id = excluded.active_validation_id,
            status = excluded.status,
            updated_at = excluded.updated_at
        """,
        (
            competition_id,
            config.competition.platform,
            config.competition.slug,
            config.competition.title,
            config.task.type.value,
            config.task.target,
            config.metric.name,
            config.metric.direction.value,
            config.validation.active,
            config.competition.status,
            now,
            now,
        ),
    )


def _upsert_validation(
    connection: sqlite3.Connection,
    competition_id: str,
    spec: ValidationSpec,
    spec_path: Path,
    comparison_domain_hash: str,
    spec_hash: str,
) -> None:
    connection.execute(
        """
        INSERT INTO validation_versions(
            id, competition_id, name, parent_id, status,
            comparison_domain_hash, spec_hash, spec_path,
            reason, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            status = excluded.status,
            comparison_domain_hash = excluded.comparison_domain_hash,
            spec_hash = excluded.spec_hash,
            spec_path = excluded.spec_path,
            reason = excluded.reason
        """,
        (
            spec.id,
            competition_id,
            spec.id,
            spec.parent,
            spec.status,
            comparison_domain_hash,
            spec_hash,
            str(spec_path),
            spec.reason,
            _utc_now(),
        ),
    )


def sync_competition(path: Path, competition_id: str, config: ArenaConfig) -> None:
    with sqlite3.connect(path) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        _upsert_competition(connection, competition_id, config)


def sync_validation(
    path: Path,
    competition_id: str,
    spec: ValidationSpec,
    spec_path: Path,
    comparison_domain_hash: str,
    spec_hash: str,
) -> None:
    with sqlite3.connect(path) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        _upsert_validation(
            connection,
            competition_id,
            spec,
            spec_path,
            comparison_domain_hash,
            spec_hash,
        )


def sync_validation_activation(
    path: Path,
    competition_id: str,
    config: ArenaConfig,
    spec: ValidationSpec,
    spec_path: Path,
    comparison_domain_hash: str,
    spec_hash: str,
) -> None:
    with sqlite3.connect(path) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        _upsert_competition(connection, competition_id, config)
        _upsert_validation(
            connection,
            competition_id,
            spec,
            spec_path,
            comparison_domain_hash,
            spec_hash,
        )
