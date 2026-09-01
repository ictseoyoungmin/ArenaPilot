import json

from typer.testing import CliRunner

from arenapilot.cli import app


runner = CliRunner()


def test_init_and_status_json(tmp_path, monkeypatch) -> None:
    destination = tmp_path / "titanic"
    result = runner.invoke(app, ["init", "kaggle:titanic", "--path", str(destination), "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["ok"] is True
    assert payload["status"] == "draft"

    monkeypatch.chdir(destination / "src")
    result = runner.invoke(app, ["status", "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["competition"] == "kaggle:titanic"
    assert payload["competition_status"] == "draft"
    assert payload["active_validation"] is None


def test_init_does_not_overwrite_existing_directory(tmp_path) -> None:
    destination = tmp_path / "existing"
    destination.mkdir()
    result = runner.invoke(app, ["init", "kaggle:titanic", "--path", str(destination), "--json"])
    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["ok"] is False
    assert payload["error"]["code"] == "WORKSPACE_INIT_FAILED"


def test_experiment_cli_create_freeze_and_list(tmp_path, monkeypatch) -> None:
    destination = tmp_path / "demo"
    assert runner.invoke(app, ["init", "kaggle:demo", "--path", str(destination)]).exit_code == 0
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
            "stratified_kfold",
            "--prediction",
            "probability",
        ],
    )
    assert result.exit_code == 0, result.output
    assert runner.invoke(app, ["validation", "activate", "val-v1"]).exit_code == 0

    result = runner.invoke(
        app,
        [
            "exp",
            "new",
            "--title",
            "baseline",
            "--hypothesis",
            "A baseline establishes the comparison floor.",
            "--model-family",
            "catboost",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["experiment"] == "exp001"
    assert payload["status"] == "draft"
    assert payload["validation"] == "val-v1"

    result = runner.invoke(app, ["exp", "freeze", "exp001", "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["status"] == "frozen"
    assert payload["config_hash"]

    result = runner.invoke(app, ["exp", "list", "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["experiments"][0]["id"] == "exp001"
    assert payload["experiments"][0]["spec_integrity"] is True
