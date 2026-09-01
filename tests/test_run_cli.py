import json

from typer.testing import CliRunner

from arenapilot.cli import app


runner = CliRunner()


def test_exp_run_cli_creates_verified_run(tmp_path, monkeypatch) -> None:
    destination = tmp_path / "demo"
    assert runner.invoke(app, ["init", "kaggle:demo", "--path", str(destination)]).exit_code == 0
    monkeypatch.chdir(destination)

    assert runner.invoke(app, ["intake", "set", "--task", "binary_classification", "--target", "target", "--metric", "roc_auc", "--direction", "maximize"]).exit_code == 0
    assert runner.invoke(app, ["validation", "configure", "val-v1", "--split", "stratified_kfold", "--prediction", "probability"]).exit_code == 0
    assert runner.invoke(app, ["validation", "activate", "val-v1"]).exit_code == 0
    assert runner.invoke(app, ["exp", "new", "--title", "baseline", "--hypothesis", "A baseline establishes the comparison floor.", "--model-family", "catboost"]).exit_code == 0
    assert runner.invoke(app, ["exp", "freeze", "exp001"]).exit_code == 0

    (destination / "src" / "train.py").write_text(
        r'''
import json
import os
from pathlib import Path
out = Path(os.environ["ARENA_OUTPUT_DIR"])
(out / "predictions.parquet").write_bytes(b"pred")
(out / "oof.parquet").write_bytes(b"oof")
(out / "result.json").write_text(json.dumps({"status": "success", "primary_metric": {"name": "roc_auc", "value": 0.82}}), encoding="utf-8")
(out / "metrics.json").write_text(json.dumps({"roc_auc": 0.82}), encoding="utf-8")
(out / "fold_metrics.json").write_text(json.dumps({"folds": [{"fold": i, "roc_auc": 0.82} for i in range(5)]}), encoding="utf-8")
print("cli trainer ok")
''',
        encoding="utf-8",
    )

    result = runner.invoke(app, ["exp", "run", "exp001", "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["run"] == "run001"
    assert payload["status"] == "verified"
    assert payload["manifest_hash"]

    result = runner.invoke(app, ["run", "show", "run001", "--json"])
    assert result.exit_code == 0, result.output
    shown = json.loads(result.output)
    assert shown["record"]["experiment_name"] == "exp001"
    assert shown["manifest"]["status"] == "verified"
