import sqlite3

from arenapilot.db import SCHEMA_VERSION, initialize_database, read_schema_version


def _tables(path):
    with sqlite3.connect(path) as connection:
        return {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }


def test_schema_v1_migrates_through_submissions_v4(tmp_path) -> None:
    path = tmp_path / "arena.db"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE schema_meta (version INTEGER NOT NULL)")
        connection.execute("INSERT INTO schema_meta(version) VALUES (1)")

    initialize_database(path)

    assert read_schema_version(path) == SCHEMA_VERSION == 4
    tables = _tables(path)
    assert "artifact_refs" in tables
    assert "remote_jobs" in tables
    assert "submissions" in tables
    assert "runs" in tables


def test_schema_v2_migrates_through_submissions_v4(tmp_path) -> None:
    path = tmp_path / "arena.db"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE schema_meta (version INTEGER NOT NULL)")
        connection.execute("INSERT INTO schema_meta(version) VALUES (2)")

    initialize_database(path)

    assert read_schema_version(path) == 4
    tables = _tables(path)
    assert "remote_jobs" in tables
    assert "submissions" in tables


def test_schema_v3_migrates_to_submissions_v4(tmp_path) -> None:
    path = tmp_path / "arena.db"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE schema_meta (version INTEGER NOT NULL)")
        connection.execute("INSERT INTO schema_meta(version) VALUES (3)")

    initialize_database(path)

    assert read_schema_version(path) == 4
    assert "submissions" in _tables(path)
