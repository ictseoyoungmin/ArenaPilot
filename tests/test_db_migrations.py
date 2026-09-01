import sqlite3

from arenapilot.db import SCHEMA_VERSION, initialize_database, read_schema_version


def test_schema_v1_migrates_to_artifact_index_v2(tmp_path) -> None:
    path = tmp_path / "arena.db"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE schema_meta (version INTEGER NOT NULL)")
        connection.execute("INSERT INTO schema_meta(version) VALUES (1)")

    initialize_database(path)

    assert read_schema_version(path) == SCHEMA_VERSION == 2
    with sqlite3.connect(path) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
    assert "artifact_refs" in tables
    assert "runs" in tables
