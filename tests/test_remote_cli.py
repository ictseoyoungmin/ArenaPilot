from __future__ import annotations

import json
import subprocess

from typer.testing import CliRunner

import arenapilot.kaggle_backend as kb
from arenapilot.cli_entry import app
from arenapilot.experiments import create_experiment, freeze_experiment
from arenapilot.intake import configure_intake
from arenapilot.models import MetricDirection, PredictionType, SplitType, TaskType
from arenapilot.validation import activate_validation, configure_validation
from arenapilot.workspace import create_workspace


runner = CliRunner()


def _ready_experiment(tmp_path):
    workspace = create_workspace("kaggle:demo", tmp_path / "demo")
    configure_intake(
        workspace,
        task_type=TaskType.BINARY_CLASSIFICATION,
        target="target",
        metric_name="roc_auc",
        metric_direction=MetricDirection.MAXIMIZE,
    )
    configure_validation(
        workspace,
        "val-v1",
        split_type=SplitType.STRATIFIED_KFOLD,
        prediction_type=PredictionType.PROBABILITY,
    )
    activate_validation(workspace, "val-v1")
    spec = create_experiment(
        workspace,
        title="baseline",
        hypothesis="A baseline establishes the comparison floor.",
        model_family="catboost",
    )
    freeze_experiment(workspace, spec.id)
    (workspace.root / "src" / "train.py").write_text("print('remote')\n", encoding="utf-8")
    return workspace, spec


def test_exp_run_dispatches_to_kaggle_and_remote_status_is_json(tmp_path, monkeypatch) -> None:
    workspace, spec = _ready_experiment(tmp_path)
    monkeypatch.chdir(workspace.root)
    monkeypatch.setenv("ARENA_KAGGLE_OWNER", "arena-user")

    def fake_cli(args: list[str]):
        if args[:2] == ["kernels", "push"]:
            return subprocess.CompletedProcess(args, 0, stdout="pushed", stderr="")
        if args[:2] == ["kernels", "status"]:
            return subprocess.CompletedProcess(
                args,
                0,
                stdout='kernel has status "running"',
                stderr="",
            )
        raise AssertionError(args)

    monkeypatch.setattr(kb, "_run_kaggle_cli", fake_cli)

    submitted = runner.invoke(
        app,
        ["exp", "run", spec.id, "--backend", "kaggle", "--json"],
    )
    assert submitted.exit_code == 0, submitted.output
    submitted_payload = json.loads(submitted.output)
    assert submitted_payload["backend"] == "kaggle"
    assert submitted_payload["status"] == "queued"
    assert submitted_payload["run"] == "run001"

    status = runner.invoke(app, ["remote", "status", "run001", "--json"])
    assert status.exit_code == 0, status.output
    status_payload = json.loads(status.output)
    assert status_payload["provider_status"] == "running"
    assert status_payload["run"]["status"] == "running"
    assert status_payload["remote_job"]["provider"] == "kaggle"
