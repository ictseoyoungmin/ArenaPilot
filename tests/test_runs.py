import pytest

from arenapilot.experiments import create_experiment, freeze_experiment
from arenapilot.intake import configure_intake
from arenapilot.models import MetricDirection, PredictionType, SplitType, TaskType
from arenapilot.runs import RunError, list_run_summaries, run_local_experiment, show_run
from arenapilot.validation import activate_validation, configure_validation
from arenapilot.workspace import create_workspace


def _ready_experiment(tmp_path, *, freeze: bool = True):
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
    if freeze:
        freeze_experiment(workspace, spec.id)
    return workspace, spec


def _write_training_module(workspace, body: str) -> None:
    (workspace.root / "src" / "train.py").write_text(body, encoding="utf-8")


GOOD_TRAINER = r'''
import json
import os
from pathlib import Path

out = Path(os.environ["ARENA_OUTPUT_DIR"])
(out / "predictions.parquet").write_bytes(b"prediction-bytes")
(out / "oof.parquet").write_bytes(b"oof-bytes")
(out / "result.json").write_text(json.dumps({
    "status": "success",
    "primary_metric": {"name": "roc_auc", "value": 0.8123}
}), encoding="utf-8")
(out / "metrics.json").write_text(json.dumps({"roc_auc": 0.8123}), encoding="utf-8")
(out / "fold_metrics.json").write_text(json.dumps({
    "folds": [{"fold": i, "roc_auc": 0.81 + i * 0.001} for i in range(5)]
}), encoding="utf-8")
print("training complete")
'''


def test_local_run_reaches_verified_and_writes_manifest(tmp_path) -> None:
    workspace, spec = _ready_experiment(tmp_path)
    _write_training_module(workspace, GOOD_TRAINER)
    record = run_local_experiment(workspace, spec.id)
    assert record["name"] == "run001"
    assert record["status"] == "verified"
    assert record["exit_code"] == 0
    assert record["artifact_manifest_hash"]

    shown = show_run(workspace, "run001")
    manifest = shown["manifest"]
    assert manifest["status"] == "verified"
    paths = {item["path"] for item in manifest["files"]}
    assert {"spec.yaml", "validation.yaml", "result.json", "metrics.json", "oof.parquet", "predictions.parquet"} <= paths


def test_every_execution_creates_a_distinct_run(tmp_path) -> None:
    workspace, spec = _ready_experiment(tmp_path)
    _write_training_module(workspace, GOOD_TRAINER)
    first = run_local_experiment(workspace, spec.id)
    second = run_local_experiment(workspace, spec.id)
    assert first["name"] == "run001"
    assert second["name"] == "run002"
    assert [item["id"] for item in list_run_summaries(workspace)] == ["run001", "run002"]


def test_nonzero_training_process_is_failed_run(tmp_path) -> None:
    workspace, spec = _ready_experiment(tmp_path)
    _write_training_module(workspace, "raise SystemExit(7)\n")
    with pytest.raises(RunError, match="exit code 7"):
        run_local_experiment(workspace, spec.id)
    summary = list_run_summaries(workspace)[0]
    assert summary["status"] == "failed"
    assert summary["exit_code"] == 7


def test_successful_process_with_missing_artifacts_is_invalid(tmp_path) -> None:
    workspace, spec = _ready_experiment(tmp_path)
    _write_training_module(
        workspace,
        r'''
import json
import os
from pathlib import Path
out = Path(os.environ["ARENA_OUTPUT_DIR"])
(out / "result.json").write_text(json.dumps({"status": "success", "primary_metric": {"name": "roc_auc", "value": 0.8}}), encoding="utf-8")
(out / "metrics.json").write_text(json.dumps({"roc_auc": 0.8}), encoding="utf-8")
''',
    )
    with pytest.raises(RunError, match="missing required artifacts"):
        run_local_experiment(workspace, spec.id)
    shown = show_run(workspace, "run001")
    assert shown["record"]["status"] == "invalid"
    assert shown["manifest"]["status"] == "invalid"


def test_draft_experiment_cannot_run(tmp_path) -> None:
    workspace, spec = _ready_experiment(tmp_path, freeze=False)
    _write_training_module(workspace, GOOD_TRAINER)
    with pytest.raises(RunError, match="EXPERIMENT_NOT_FROZEN"):
        run_local_experiment(workspace, spec.id)
    assert list_run_summaries(workspace) == []
