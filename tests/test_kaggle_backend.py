from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

import arenapilot.kaggle_backend as kb
from arenapilot.experiments import create_experiment, freeze_experiment
from arenapilot.intake import configure_intake
from arenapilot.models import MetricDirection, PredictionType, SplitType, TaskType
from arenapilot.remote_store import get_remote_job_for_run
from arenapilot.runstore import get_run
from arenapilot.validation import activate_validation, configure_validation
from arenapilot.workspace import create_workspace


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
    (workspace.root / "src" / "train.py").write_text(
        "print('remote trainer source')\n",
        encoding="utf-8",
    )
    return workspace, spec


def _completed(args, stdout: str = "", stderr: str = ""):
    return subprocess.CompletedProcess(args=args, returncode=0, stdout=stdout, stderr=stderr)


def _write_remote_artifacts(workspace, spec, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "spec.yaml").write_text(
        workspace.experiment_path(spec.id).read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (output / "validation.yaml").write_text(
        workspace.validation_path("val-v1").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (output / "environment.json").write_text(
        json.dumps({"schema_version": 1, "backend": "kaggle"}),
        encoding="utf-8",
    )
    (output / "logs.txt").write_text("remote training complete\n", encoding="utf-8")
    (output / "predictions.parquet").write_bytes(b"predictions")
    (output / "oof.parquet").write_bytes(b"oof")
    (output / "result.json").write_text(
        json.dumps(
            {
                "status": "success",
                "primary_metric": {"name": "roc_auc", "value": 0.834},
            }
        ),
        encoding="utf-8",
    )
    (output / "metrics.json").write_text(
        json.dumps({"roc_auc": 0.834, "cv_std": 0.002}),
        encoding="utf-8",
    )
    (output / "fold_metrics.json").write_text(
        json.dumps(
            {
                "folds": [
                    {"fold": index, "roc_auc": 0.832 + index * 0.001}
                    for index in range(5)
                ]
            }
        ),
        encoding="utf-8",
    )


def test_kaggle_run_pushes_bundle_and_records_remote_job(tmp_path, monkeypatch) -> None:
    workspace, spec = _ready_experiment(tmp_path)
    calls: list[list[str]] = []

    def fake_cli(args: list[str]):
        calls.append(args)
        assert args[:2] == ["kernels", "push"]
        return _completed(args, stdout="Kernel version successfully pushed")

    monkeypatch.setattr(kb, "_run_kaggle_cli", fake_cli)
    record = kb.run_kaggle_experiment(workspace, spec.id, owner="arena-user")

    assert record["name"] == "run001"
    assert record["backend"] == "kaggle"
    assert record["status"] == "queued"
    assert calls and calls[0][0:2] == ["kernels", "push"]

    job = get_remote_job_for_run(workspace.db_path, str(record["id"]))
    assert job is not None
    assert job["provider"] == "kaggle"
    assert str(job["provider_job_id"]).startswith("arena-user/arenapilot-demo-run001")
    assert job["state"] == "submitted"
    assert job["recovery_state"] == "pending"

    bundle = Path(str(job["bundle_path"]))
    metadata = json.loads((bundle / "kernel-metadata.json").read_text(encoding="utf-8"))
    assert metadata["id"] == job["provider_job_id"]
    assert metadata["competition_sources"] == ["demo"]
    assert metadata["kernel_type"] == "script"
    runner = (bundle / "arena_kernel.py").read_text(encoding="utf-8")
    assert "ARENA_DATA_DIR" in runner
    assert "remote trainer source" not in runner  # source is encoded, not interpolated raw


def test_completed_kaggle_run_recovers_through_normal_verification(tmp_path, monkeypatch) -> None:
    workspace, spec = _ready_experiment(tmp_path)

    def fake_cli(args: list[str]):
        if args[:2] == ["kernels", "push"]:
            return _completed(args, stdout="pushed")
        if args[:2] == ["kernels", "status"]:
            return _completed(args, stdout='kernel has status "complete"')
        if args[:2] == ["kernels", "output"]:
            output = Path(args[args.index("-p") + 1])
            _write_remote_artifacts(workspace, spec, output)
            return _completed(args, stdout="downloaded")
        raise AssertionError(args)

    monkeypatch.setattr(kb, "_run_kaggle_cli", fake_cli)
    queued = kb.run_kaggle_experiment(workspace, spec.id, owner="arena-user")
    verified = kb.recover_kaggle_run(workspace, str(queued["name"]))

    assert verified["status"] == "verified"
    assert verified["backend"] == "kaggle"
    assert verified["mlflow_run_id"]
    assert (_run_dir := workspace.root / "outputs" / "runs" / "run001").is_dir()
    assert (_run_dir / "manifest.json").is_file()

    job = get_remote_job_for_run(workspace.db_path, str(verified["id"]))
    assert job is not None
    assert job["state"] == "completed"
    assert job["recovery_state"] == "verified"


def test_kaggle_status_moves_queued_run_to_running(tmp_path, monkeypatch) -> None:
    workspace, spec = _ready_experiment(tmp_path)

    def fake_cli(args: list[str]):
        if args[:2] == ["kernels", "push"]:
            return _completed(args, stdout="pushed")
        if args[:2] == ["kernels", "status"]:
            return _completed(args, stdout='kernel has status "running"')
        raise AssertionError(args)

    monkeypatch.setattr(kb, "_run_kaggle_cli", fake_cli)
    queued = kb.run_kaggle_experiment(workspace, spec.id, owner="arena-user")
    payload = kb.sync_kaggle_status(workspace, str(queued["name"]))

    assert payload["provider_status"] == "running"
    assert payload["run"]["status"] == "running"
    persisted = get_run(workspace.db_path, workspace.competition_id, "run001")
    assert persisted is not None and persisted["status"] == "running"


def test_failed_kaggle_run_preserves_provider_logs(tmp_path, monkeypatch) -> None:
    workspace, spec = _ready_experiment(tmp_path)

    def fake_cli(args: list[str]):
        if args[:2] == ["kernels", "push"]:
            return _completed(args, stdout="pushed")
        if args[:2] == ["kernels", "status"]:
            return _completed(args, stdout='kernel has status "error"')
        if args[:2] == ["kernels", "logs"]:
            return _completed(args, stdout="remote traceback\n")
        raise AssertionError(args)

    monkeypatch.setattr(kb, "_run_kaggle_cli", fake_cli)
    queued = kb.run_kaggle_experiment(workspace, spec.id, owner="arena-user")
    with pytest.raises(kb.KaggleBackendError, match="REMOTE_JOB_FAILED"):
        kb.recover_kaggle_run(workspace, str(queued["name"]))

    run = get_run(workspace.db_path, workspace.competition_id, "run001")
    assert run is not None and run["status"] == "failed"
    assert (workspace.root / "outputs" / "runs" / "run001" / "logs.txt").read_text(
        encoding="utf-8"
    ) == "remote traceback\n"


def test_kaggle_owner_is_explicit_when_credentials_do_not_expose_username(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("ARENA_KAGGLE_OWNER", raising=False)
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.setenv("KAGGLE_CONFIG_DIR", str(tmp_path / "missing-config"))
    with pytest.raises(kb.KaggleBackendError, match="KAGGLE_OWNER_REQUIRED"):
        kb.resolve_kaggle_owner()
