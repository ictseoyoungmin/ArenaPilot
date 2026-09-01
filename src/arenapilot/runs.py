from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

from .db import get_experiment
from .experiments import ExperimentError, freeze_experiment
from .models import ExperimentSpec, ValidationSpec
from .runstore import create_run_record, get_run, list_runs, transition_run
from .workspace import Workspace, WorkspaceError, load_experiment_spec


class RunError(WorkspaceError):
    pass


def _canonical_hash(payload: object) -> str:
    raw = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RunError(f"invalid JSON artifact: {path.name}") from exc


def _run_dir(workspace: Workspace, name: str) -> Path:
    return workspace.root / "outputs" / "runs" / name


def _manifest_for_directory(
    run_dir: Path,
    *,
    run_name: str,
    status: str,
    verification_error: str | None = None,
) -> tuple[Path, str]:
    files: list[dict[str, object]] = []
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file() or path.name == "manifest.json":
            continue
        files.append(
            {
                "path": str(path.relative_to(run_dir)),
                "sha256": _sha256_file(path),
                "size": path.stat().st_size,
            }
        )
    payload = {
        "schema_version": 1,
        "run": run_name,
        "status": status,
        "verification_error": verification_error,
        "files": files,
    }
    manifest_path = run_dir / "manifest.json"
    _write_json(manifest_path, payload)
    return manifest_path, _sha256_file(manifest_path)


def _load_run_specs(run_dir: Path) -> tuple[ExperimentSpec, ValidationSpec]:
    try:
        experiment_raw = yaml.safe_load((run_dir / "spec.yaml").read_text(encoding="utf-8"))
        validation_raw = yaml.safe_load((run_dir / "validation.yaml").read_text(encoding="utf-8"))
    except OSError as exc:
        raise RunError("run spec or validation snapshot is missing") from exc
    return (
        ExperimentSpec.model_validate(experiment_raw),
        ValidationSpec.model_validate(validation_raw),
    )


def _validate_artifacts(run_dir: Path, row: dict[str, object]) -> None:
    required = {
        "spec.yaml",
        "validation.yaml",
        "environment.json",
        "result.json",
        "metrics.json",
        "predictions.parquet",
        "logs.txt",
    }
    experiment, validation = _load_run_specs(run_dir)
    if validation.oof.required:
        required.update({"oof.parquet", "fold_metrics.json"})

    missing = sorted(name for name in required if not (run_dir / name).is_file())
    if missing:
        raise RunError("missing required artifacts: " + ", ".join(missing))

    if experiment.id != row["experiment_name"]:
        raise RunError("run experiment snapshot does not match the persisted run")
    if experiment.validation != validation.id:
        raise RunError("run validation snapshot does not match the experiment")
    if validation.metric is None:
        raise RunError("run validation metric is incomplete")

    spec_hash = _canonical_hash(experiment.model_dump(mode="json"))
    if spec_hash != row["spec_hash"]:
        raise RunError("run experiment snapshot hash does not match the frozen experiment")

    result = _read_json(run_dir / "result.json")
    metrics = _read_json(run_dir / "metrics.json")
    if not isinstance(result, dict) or result.get("status") != "success":
        raise RunError("result.json must declare status=success")
    if not isinstance(metrics, dict):
        raise RunError("metrics.json must contain an object")

    primary = result.get("primary_metric")
    if not isinstance(primary, dict):
        raise RunError("result.json must contain primary_metric")
    if primary.get("name") != validation.metric.name:
        raise RunError("primary metric name does not match the validation contract")
    value = primary.get("value")
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
        raise RunError("primary metric value must be finite")

    metric_value = metrics.get(validation.metric.name)
    if not isinstance(metric_value, (int, float)) or isinstance(metric_value, bool) or not math.isfinite(float(metric_value)):
        raise RunError(f"metrics.json must contain finite metric {validation.metric.name}")

    for artifact in ("predictions.parquet", "oof.parquet"):
        path = run_dir / artifact
        if path.is_file() and path.stat().st_size == 0:
            raise RunError(f"{artifact} must not be empty")

    if validation.oof.required:
        fold_metrics = _read_json(run_dir / "fold_metrics.json")
        folds = fold_metrics.get("folds") if isinstance(fold_metrics, dict) else None
        if not isinstance(folds, list) or len(folds) != validation.split.n_splits:
            raise RunError(
                f"fold_metrics.json must contain {validation.split.n_splits} folds"
            )


def verify_run(workspace: Workspace, name: str) -> dict[str, object]:
    row = get_run(workspace.db_path, workspace.competition_id, name)
    if row is None:
        raise RunError(f"run not found: {name}")
    if row["status"] == "verified":
        return row
    if row["status"] not in {"completed", "invalid"}:
        raise RunError(f"run cannot be verified from status {row['status']}")

    transition_run(
        workspace.db_path,
        competition_id=workspace.competition_id,
        name=name,
        from_statuses={str(row["status"])},
        to_status="verifying",
    )
    run_dir = _run_dir(workspace, name)
    try:
        _validate_artifacts(run_dir, row)
        manifest_path, manifest_hash = _manifest_for_directory(
            run_dir,
            run_name=name,
            status="verified",
        )
    except Exception as exc:
        manifest_path, manifest_hash = _manifest_for_directory(
            run_dir,
            run_name=name,
            status="invalid",
            verification_error=str(exc),
        )
        transition_run(
            workspace.db_path,
            competition_id=workspace.competition_id,
            name=name,
            from_statuses={"verifying"},
            to_status="invalid",
            manifest_path=str(manifest_path),
            manifest_hash=manifest_hash,
        )
        if isinstance(exc, RunError):
            raise
        raise RunError(str(exc)) from exc

    return transition_run(
        workspace.db_path,
        competition_id=workspace.competition_id,
        name=name,
        from_statuses={"verifying"},
        to_status="verified",
        manifest_path=str(manifest_path),
        manifest_hash=manifest_hash,
    )


def run_local_experiment(
    workspace: Workspace,
    experiment_name: str,
    *,
    backend: str | None = None,
) -> dict[str, object]:
    experiment_row = get_experiment(
        workspace.db_path,
        workspace.competition_id,
        experiment_name,
    )
    if experiment_row is None:
        raise RunError(f"experiment not found: {experiment_name}")
    if experiment_row["status"] != "frozen":
        raise RunError("EXPERIMENT_NOT_FROZEN")

    try:
        _, snapshot = freeze_experiment(workspace, experiment_name)
    except ExperimentError as exc:
        raise RunError(str(exc)) from exc

    spec = load_experiment_spec(workspace, experiment_name)
    selected_backend = backend or spec.runtime.backend
    if selected_backend != "local":
        raise RunError(f"backend not supported by local runner: {selected_backend}")

    record = create_run_record(
        workspace.db_path,
        competition_id=workspace.competition_id,
        experiment_id=str(experiment_row["id"]),
        backend="local",
        spec_hash=str(experiment_row["config_hash"]),
    )
    run_name = str(record["name"])
    run_dir = _run_dir(workspace, run_name)
    run_dir.mkdir(parents=True, exist_ok=False)

    shutil.copy2(snapshot, run_dir / "spec.yaml")
    validation_source = workspace.validation_path(spec.validation)
    if not validation_source.is_file():
        raise RunError(f"validation snapshot source missing: {spec.validation}")
    shutil.copy2(validation_source, run_dir / "validation.yaml")
    _write_json(
        run_dir / "environment.json",
        {
            "schema_version": 1,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "backend": "local",
            "executable": sys.executable,
        },
    )

    transition_run(
        workspace.db_path,
        competition_id=workspace.competition_id,
        name=run_name,
        from_statuses={"created"},
        to_status="running",
    )

    env = os.environ.copy()
    env.update(
        {
            "ARENA_RUN_ID": run_name,
            "ARENA_EXPERIMENT_SPEC": str((run_dir / "spec.yaml").resolve()),
            "ARENA_VALIDATION_SPEC": str((run_dir / "validation.yaml").resolve()),
            "ARENA_OUTPUT_DIR": str(run_dir.resolve()),
            "ARENA_DATA_DIR": str((workspace.root / "data" / "raw").resolve()),
        }
    )

    try:
        process = subprocess.run(
            [sys.executable, "-m", "src.train"],
            cwd=workspace.root,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception as exc:
        (run_dir / "logs.txt").write_text(str(exc) + "\n", encoding="utf-8")
        transition_run(
            workspace.db_path,
            competition_id=workspace.competition_id,
            name=run_name,
            from_statuses={"running"},
            to_status="failed",
            exit_code=-1,
        )
        raise RunError(f"local runner failed to start: {exc}") from exc

    log_text = process.stdout
    if process.stderr:
        if log_text and not log_text.endswith("\n"):
            log_text += "\n"
        log_text += process.stderr
    (run_dir / "logs.txt").write_text(log_text, encoding="utf-8")

    if process.returncode != 0:
        failed = transition_run(
            workspace.db_path,
            competition_id=workspace.competition_id,
            name=run_name,
            from_statuses={"running"},
            to_status="failed",
            exit_code=process.returncode,
        )
        raise RunError(f"run {run_name} failed with exit code {failed['exit_code']}")

    transition_run(
        workspace.db_path,
        competition_id=workspace.competition_id,
        name=run_name,
        from_statuses={"running"},
        to_status="completed",
        exit_code=0,
    )
    return verify_run(workspace, run_name)


def show_run(workspace: Workspace, name: str) -> dict[str, object]:
    row = get_run(workspace.db_path, workspace.competition_id, name)
    if row is None:
        raise RunError(f"run not found: {name}")
    manifest: object | None = None
    manifest_path = row.get("artifact_manifest_path")
    if manifest_path and Path(str(manifest_path)).is_file():
        manifest = _read_json(Path(str(manifest_path)))
    return {"record": row, "manifest": manifest}


def list_run_summaries(workspace: Workspace) -> list[dict[str, object]]:
    return [
        {
            "id": row["name"],
            "experiment": row["experiment_name"],
            "status": row["status"],
            "backend": row["backend"],
            "exit_code": row["exit_code"],
            "manifest_hash": row["artifact_manifest_hash"],
        }
        for row in list_runs(workspace.db_path, workspace.competition_id)
    ]


def run_logs(workspace: Workspace, name: str) -> str:
    if get_run(workspace.db_path, workspace.competition_id, name) is None:
        raise RunError(f"run not found: {name}")
    path = _run_dir(workspace, name) / "logs.txt"
    if not path.is_file():
        raise RunError(f"logs not found for run: {name}")
    return path.read_text(encoding="utf-8")
