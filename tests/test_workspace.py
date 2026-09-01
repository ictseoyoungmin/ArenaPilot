import json

import pytest
import yaml

from arenapilot.db import SCHEMA_VERSION, read_schema_version
from arenapilot.workspace import (
    WorkspaceExistsError,
    WorkspaceNotFoundError,
    create_workspace,
    discover_workspace,
    load_arena_config,
    parse_competition_ref,
)


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


def test_create_workspace_builds_draft_scaffold(tmp_path) -> None:
    destination = tmp_path / "titanic"
    workspace = create_workspace("kaggle:titanic", destination, title="Titanic")

    assert workspace.root == destination
    assert (destination / "arena.yaml").is_file()
    assert (destination / "configs" / "validation" / "val-v1.yaml").is_file()
    assert (destination / "outputs" / "runs").is_dir()
    assert (destination / ".arena" / "workspace.json").is_file()
    assert read_schema_version(workspace.db_path) == SCHEMA_VERSION

    config = load_arena_config(workspace)
    assert config.competition.slug == "titanic"
    assert config.competition.status == "draft"
    assert config.task is None
    assert config.metric is None
    assert config.validation.active is None

    validation = yaml.safe_load(
        (destination / "configs" / "validation" / "val-v1.yaml").read_text(encoding="utf-8")
    )
    assert validation["status"] == "draft"
    assert validation["metric"] is None
    assert validation["prediction"] is None


def test_create_workspace_refuses_to_overwrite_existing_destination(tmp_path) -> None:
    destination = tmp_path / "existing"
    destination.mkdir()
    sentinel = destination / "keep.txt"
    sentinel.write_text("keep", encoding="utf-8")

    with pytest.raises(WorkspaceExistsError):
        create_workspace("kaggle:titanic", destination)

    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_competition_ref_is_explicit() -> None:
    assert parse_competition_ref("kaggle:titanic") == ("kaggle", "titanic")
    with pytest.raises(Exception):
        parse_competition_ref("titanic")
    with pytest.raises(Exception):
        parse_competition_ref("other:titanic")
