from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from arenapilot.cli_entry import app
from arenapilot.experiments import create_experiment, freeze_experiment
from arenapilot.intake import configure_intake
from arenapilot.models import MetricDirection, PredictionType, SplitType, TaskType
from arenapilot.runs import run_local_experiment
from arenapilot.submissions import (
    SubmissionError,
    budget_status,
    create_submission,
    list_submission_summaries,
    send_submission,
    sync_submission,
    validate_submission,
)
from arenapilot.validation import activate_validation, configure_validation
from arenapilot.workspace import create_workspace


runner = CliRunner()


GOOD_TRAINER_WITH_SUBMISSION = r'''
import json
import os
from pathlib import Path

out = Path(os.environ["ARENA_OUTPUT_DIR"])
(out / "predictions.parquet").write_bytes(b"prediction-bytes")
(out / "oof.parquet").write_bytes(b"oof-bytes")
(out / "submission.csv").write_text("id,target\n10,0.2\n11,0.8\n", encoding="utf-8")
(out / "result.json").write_text(json.dumps({
    "status": "success",
    "primary_metric": {"name": "roc_auc", "value": 0.8123}
}), encoding="utf-8")
(out / "metrics.json").write_text(json.dumps({"roc_auc": 0.8123}), encoding="utf-8")
(out / "fold_metrics.json").write_text(json.dumps({
    "folds": [{"fold": i, "roc_auc": 0.81 + i * 0.001} for i in range(5)]
}), encoding="utf-8")
'''


class FakeSubmissionProvider:
    def __init__(self) -> None:
        self.sent: list[tuple[str, Path, str]] = []
        self.next_ref = 1000

    def submit_file(self, competition: str, file_path: Path, message: str) -> str:
        self.sent.append((competition, file_path, message))
        self.next_ref += 1
        return str(self.next_ref)

    def submission_status(self, competition: str, submission_ref: str) -> dict[str, object]:
        return {
            "ref": submission_ref,
            "status": "complete",
            "publicScore": "0.7991",
            "privateScore": None,
        }

    def submission_limits(self, competition: str) -> dict[str, object] | None:
        return {"competition": competition, "remainingDailySubmissions": 7}


def _ready_verified_run(tmp_path):
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
    (workspace.root / "src" / "train.py").write_text(
        GOOD_TRAINER_WITH_SUBMISSION,
        encoding="utf-8",
    )
    (workspace.root / "data" / "raw" / "sample_submission.csv").write_text(
        "id,target\n10,0.5\n11,0.5\n",
        encoding="utf-8",
    )
    run = run_local_experiment(workspace, spec.id)
    assert run["status"] == "verified"
    return workspace, spec, run


def test_submission_lifecycle_reaches_scored_state(tmp_path) -> None:
    workspace, _, run = _ready_verified_run(tmp_path)
    provider = FakeSubmissionProvider()

    created = create_submission(workspace, run_name=str(run["name"]), message="baseline")
    assert created["name"] == "sub001"
    assert created["status"] == "created"
    assert Path(str(created["file_path"])).read_text(encoding="utf-8") == "id,target\n10,0.2\n11,0.8\n"

    validated = validate_submission(workspace, "sub001")
    assert validated["status"] == "validated"

    sent = send_submission(workspace, "sub001", provider=provider)
    assert sent["status"] == "submitted"
    assert sent["platform_submission_id"] == "1001"
    assert provider.sent[0][0] == "demo"
    assert provider.sent[0][2] == "baseline"

    scored = sync_submission(workspace, "sub001", provider=provider)
    assert scored["status"] == "scored"
    assert scored["public_score"] == pytest.approx(0.7991)

    summaries = list_submission_summaries(workspace)
    assert summaries == [
        {
            "id": "sub001",
            "experiment": "exp001",
            "run": "run001",
            "status": "scored",
            "platform_submission_id": "1001",
            "public_score": pytest.approx(0.7991),
            "private_score": None,
            "message": "baseline",
            "file_sha256": created["file_sha256"],
        }
    ]


def test_submission_validation_rejects_id_mismatch(tmp_path) -> None:
    workspace, _, run = _ready_verified_run(tmp_path)
    bad = workspace.root / "bad.csv"
    bad.write_text("id,target\n11,0.2\n10,0.8\n", encoding="utf-8")
    created = create_submission(workspace, run_name=str(run["name"]), file_path=bad)
    with pytest.raises(SubmissionError, match="SUBMISSION_ID_MISMATCH"):
        validate_submission(workspace, str(created["name"]))


def test_submission_validation_rejects_invalid_probability(tmp_path) -> None:
    workspace, _, run = _ready_verified_run(tmp_path)
    bad = workspace.root / "bad-probability.csv"
    bad.write_text("id,target\n10,1.2\n11,0.8\n", encoding="utf-8")
    created = create_submission(workspace, run_name=str(run["name"]), file_path=bad)
    with pytest.raises(SubmissionError, match="probability outside"):
        validate_submission(workspace, str(created["name"]))


def test_submission_budget_is_enforced_before_provider_call(tmp_path) -> None:
    workspace, _, run = _ready_verified_run(tmp_path)
    provider = FakeSubmissionProvider()

    for index in range(3):
        created = create_submission(workspace, run_name=str(run["name"]), message=f"s{index}")
        validate_submission(workspace, str(created["name"]))
        send_submission(workspace, str(created["name"]), provider=provider)

    fourth = create_submission(workspace, run_name=str(run["name"]), message="blocked")
    validate_submission(workspace, str(fourth["name"]))
    with pytest.raises(SubmissionError, match="SUBMISSION_BUDGET_EXCEEDED"):
        send_submission(workspace, str(fourth["name"]), provider=provider)
    assert len(provider.sent) == 3

    budget = budget_status(workspace, provider=provider)
    assert budget["daily_used"] == 3
    assert budget["daily_remaining"] == 0
    assert budget["provider_limits"]["remainingDailySubmissions"] == 7


def test_cli_create_and_validate_emit_machine_readable_json(tmp_path, monkeypatch) -> None:
    workspace, _, run = _ready_verified_run(tmp_path)
    monkeypatch.chdir(workspace.root)

    created_result = runner.invoke(
        app,
        ["submit", "create", "--run", str(run["name"]), "--json"],
    )
    assert created_result.exit_code == 0, created_result.output
    created = json.loads(created_result.stdout)
    assert created["ok"] is True
    assert created["submission"] == "sub001"

    validated_result = runner.invoke(
        app,
        ["submit", "validate", "sub001", "--json"],
    )
    assert validated_result.exit_code == 0, validated_result.output
    validated = json.loads(validated_result.stdout)
    assert validated == {"ok": True, "status": "validated", "submission": "sub001"}

    listed_result = runner.invoke(app, ["submissions", "--json"])
    assert listed_result.exit_code == 0, listed_result.output
    listed = json.loads(listed_result.stdout)
    assert listed["submissions"][0]["id"] == "sub001"
