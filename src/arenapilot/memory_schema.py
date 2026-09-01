from __future__ import annotations

import sqlite3
from pathlib import Path

from .db import initialize_database


MEMORY_SCHEMA_VERSION = 1

_MEMORY_SCHEMA = """
CREATE TABLE IF NOT EXISTS competition_fingerprints (
    id TEXT PRIMARY KEY,
    competition_id TEXT NOT NULL,
    schema_version INTEGER NOT NULL,
    fingerprint_json TEXT NOT NULL,
    fingerprint_hash TEXT NOT NULL,
    source TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (competition_id) REFERENCES competitions(id),
    UNIQUE(competition_id, fingerprint_hash)
);
CREATE INDEX IF NOT EXISTS idx_competition_fingerprints_competition
    ON competition_fingerprints(competition_id, created_at);

CREATE TABLE IF NOT EXISTS memory_evidence (
    id TEXT PRIMARY KEY,
    competition_id TEXT NOT NULL,
    name TEXT NOT NULL,
    subject_type TEXT NOT NULL,
    subject_key TEXT NOT NULL,
    source_experiment_id TEXT,
    source_run_id TEXT,
    source_submission_id TEXT,
    reference_experiment_id TEXT,
    reference_run_id TEXT,
    validation_domain_hash TEXT,
    outcome TEXT NOT NULL,
    effect REAL,
    strength INTEGER NOT NULL CHECK(strength >= 0 AND strength <= 3),
    summary TEXT NOT NULL,
    context_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (competition_id) REFERENCES competitions(id),
    FOREIGN KEY (source_experiment_id) REFERENCES experiments(id),
    FOREIGN KEY (source_run_id) REFERENCES runs(id),
    FOREIGN KEY (source_submission_id) REFERENCES submissions(id),
    FOREIGN KEY (reference_experiment_id) REFERENCES experiments(id),
    FOREIGN KEY (reference_run_id) REFERENCES runs(id),
    UNIQUE(competition_id, name),
    CHECK(
        source_experiment_id IS NOT NULL OR
        source_run_id IS NOT NULL OR
        source_submission_id IS NOT NULL
    )
);
CREATE INDEX IF NOT EXISTS idx_memory_evidence_subject
    ON memory_evidence(competition_id, subject_type, subject_key);

CREATE TABLE IF NOT EXISTS findings (
    id TEXT PRIMARY KEY,
    competition_id TEXT NOT NULL,
    name TEXT NOT NULL,
    subject_type TEXT NOT NULL,
    subject_key TEXT NOT NULL,
    conclusion TEXT NOT NULL,
    summary TEXT NOT NULL,
    confidence TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (competition_id) REFERENCES competitions(id),
    UNIQUE(competition_id, name)
);
CREATE INDEX IF NOT EXISTS idx_findings_subject
    ON findings(competition_id, subject_type, subject_key, status);

CREATE TABLE IF NOT EXISTS finding_evidence (
    finding_id TEXT NOT NULL,
    evidence_id TEXT NOT NULL,
    role TEXT NOT NULL,
    PRIMARY KEY (finding_id, evidence_id, role),
    FOREIGN KEY (finding_id) REFERENCES findings(id),
    FOREIGN KEY (evidence_id) REFERENCES memory_evidence(id)
);
"""


def initialize_workspace_memory_schema(path: Path) -> None:
    initialize_database(path)
    with sqlite3.connect(path) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute(
            "CREATE TABLE IF NOT EXISTS memory_schema_meta (version INTEGER NOT NULL)"
        )
        row = connection.execute(
            "SELECT version FROM memory_schema_meta LIMIT 1"
        ).fetchone()
        if row is None:
            connection.execute(
                "INSERT INTO memory_schema_meta(version) VALUES (?)",
                (MEMORY_SCHEMA_VERSION,),
            )
        elif int(row[0]) != MEMORY_SCHEMA_VERSION:
            raise RuntimeError(f"unsupported workspace memory schema version: {row[0]}")
        connection.executescript(_MEMORY_SCHEMA)


def read_workspace_memory_schema_version(path: Path) -> int:
    with sqlite3.connect(path) as connection:
        row = connection.execute(
            "SELECT version FROM memory_schema_meta LIMIT 1"
        ).fetchone()
    if row is None:
        raise RuntimeError("workspace memory schema version is missing")
    return int(row[0])
