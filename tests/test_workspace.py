import json

import pytest

from arenapilot.workspace import WorkspaceNotFoundError, discover_workspace


def test_workspace_discovery_walks_parents(tmp_path) -> None:
    root = tmp_path / "competition"
    nested = root / "src" / "models"
    marker = root / ".arena" / "workspace.json"
    marker.parent.mkdir(parents=True)
    nested.mkdir(parents=True)
    marker.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "competition_id": "cmp_01TEST",
                "workspace_id": "ws_01TEST",
            }
        ),
        encoding="utf-8",
    )

    workspace = discover_workspace(nested)
    assert workspace.root == root
    assert workspace.db_path == root / ".arena" / "arena.db"


def test_workspace_discovery_fails_outside_workspace(tmp_path) -> None:
    with pytest.raises(WorkspaceNotFoundError):
        discover_workspace(tmp_path)
