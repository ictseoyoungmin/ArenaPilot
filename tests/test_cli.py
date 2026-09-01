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
