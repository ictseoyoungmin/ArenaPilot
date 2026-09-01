import json
import sqlite3

from typer.testing import CliRunner

from arenapilot.cli import app
from arenapilot.workspace import (
    discover_workspace,
    load_arena_config,
    load_validation_spec,
)


runner = CliRunner()


def _init_workspace(tmp_path):
    destination = tmp_path / "demo"
    result = runner.invoke(
        app,
        ["init", "kaggle:demo", "--path", str(destination), "--json"],
    )
    assert result.exit_code == 0, result.output
    return destination


def test_intake_validation_activation_end_to_end(tmp_path, monkeypatch) -> None:
    destination = _init_workspace(tmp_path)
    monkeypatch.chdir(destination)

    result = runner.invoke(
        app,
        [
            "intake",
            "set",
            "--task",
            "binary_classification",
            "--target",
            "target",
            "--metric",
            "roc_auc",
            "--direction",
            "maximize",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["competition_status"] == "draft"

    result = runner.invoke(
        app,
        [
            "validation",
            "configure",
            "val-v1",
            "--split",
            "stratified_kfold",
            "--prediction",
            "probability",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output

    result = runner.invoke(
        app,
        ["validation", "activate", "val-v1", "--json"],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["competition_status"] == "ready"
    assert payload["active_validation"] == "val-v1"

    workspace = discover_workspace()
    config = load_arena_config(workspace)
    validation = load_validation_spec(workspace, "val-v1")
    assert config.competition.status == "ready"
    assert validation.status == "active"

    with sqlite3.connect(workspace.db_path) as connection:
        competition = connection.execute(
            "SELECT status, active_validation_id FROM competitions WHERE id = ?",
            (workspace.competition_id,),
        ).fetchone()
        validation_row = connection.execute(
            """
            SELECT status, comparison_domain_hash, spec_hash
            FROM validation_versions
            WHERE id = ?
            """,
            ("val-v1",),
        ).fetchone()

    assert competition == ("ready", "val-v1")
    assert validation_row is not None
    assert validation_row[0] == "active"
    assert len(validation_row[1]) == 64
    assert len(validation_row[2]) == 64


def test_activation_rejects_unconfigured_validation(tmp_path, monkeypatch) -> None:
    destination = _init_workspace(tmp_path)
    monkeypatch.chdir(destination)

    result = runner.invoke(
        app,
        [
            "intake",
            "set",
            "--task",
            "regression",
            "--target",
            "y",
            "--metric",
            "rmse",
            "--direction",
            "minimize",
        ],
    )
    assert result.exit_code == 0, result.output

    result = runner.invoke(
        app,
        ["validation", "activate", "val-v1", "--json"],
    )
    assert result.exit_code == 1
    assert json.loads(result.output)["error"]["code"] == "VALIDATION_ACTIVATION_FAILED"


def test_regression_rejects_probability_prediction(tmp_path, monkeypatch) -> None:
    destination = _init_workspace(tmp_path)
    monkeypatch.chdir(destination)

    result = runner.invoke(
        app,
        [
            "intake",
            "set",
            "--task",
            "regression",
            "--target",
            "y",
            "--metric",
            "rmse",
            "--direction",
            "minimize",
        ],
    )
    assert result.exit_code == 0, result.output

    result = runner.invoke(
        app,
        [
            "validation",
            "configure",
            "val-v1",
            "--split",
            "kfold",
            "--prediction",
            "probability",
            "--json",
        ],
    )
    assert result.exit_code == 1
    assert json.loads(result.output)["error"]["code"] == "VALIDATION_CONFIG_FAILED"


def test_ready_competition_intake_is_immutable(tmp_path, monkeypatch) -> None:
    destination = _init_workspace(tmp_path)
    monkeypatch.chdir(destination)

    assert runner.invoke(
        app,
        [
            "intake",
            "set",
            "--task",
            "regression",
            "--target",
            "y",
            "--metric",
            "rmse",
            "--direction",
            "minimize",
        ],
    ).exit_code == 0
    assert runner.invoke(
        app,
        [
            "validation",
            "configure",
            "val-v1",
            "--split",
            "kfold",
            "--prediction",
            "value",
        ],
    ).exit_code == 0
    assert runner.invoke(app, ["validation", "activate", "val-v1"]).exit_code == 0

    result = runner.invoke(
        app,
        [
            "intake",
            "set",
            "--task",
            "regression",
            "--target",
            "z",
            "--metric",
            "mae",
            "--direction",
            "minimize",
            "--json",
        ],
    )
    assert result.exit_code == 1
    assert json.loads(result.output)["error"]["code"] == "INTAKE_CONFIG_FAILED"
