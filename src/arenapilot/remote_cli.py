from __future__ import annotations

import json

import typer

from .kaggle_backend import (
    KaggleBackendError,
    kaggle_remote_logs,
    recover_kaggle_run,
    sync_kaggle_status,
)
from .workspace import WorkspaceError, discover_workspace


remote_app = typer.Typer(
    no_args_is_help=True,
    help="Inspect and recover remote execution jobs.",
)


def _fail(code: str, exc: Exception, json_output: bool) -> None:
    if json_output:
        typer.echo(
            json.dumps(
                {
                    "ok": False,
                    "error": {"code": code, "message": str(exc)},
                }
            )
        )
    else:
        typer.echo(f"ArenaPilot failed: {exc}", err=True)
    raise typer.Exit(code=1)


@remote_app.command("status")
def remote_status(
    run_name: str = typer.Argument(..., help="Kaggle-backed Run ID, e.g. run001."),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    """Synchronize the latest provider state for a remote Run."""
    try:
        workspace = discover_workspace()
        payload = {"ok": True, **sync_kaggle_status(workspace, run_name)}
    except (KaggleBackendError, WorkspaceError, ValueError, OSError) as exc:
        _fail("REMOTE_STATUS_FAILED", exc, json_output)

    if json_output:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
    else:
        run = payload["run"]
        job = payload["remote_job"]
        typer.echo(f"Run: {run['name']} ({run['status']})")
        typer.echo(f"Provider: {job['provider']}:{job['provider_job_id']}")
        typer.echo(f"Remote state: {payload['provider_status']}")
        typer.echo(f"Recovery: {job['recovery_state']}")


@remote_app.command("recover")
def remote_recover(
    run_name: str = typer.Argument(..., help="Completed Kaggle-backed Run ID."),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    """Pull completed remote artifacts and pass them through normal verification."""
    try:
        workspace = discover_workspace()
        record = recover_kaggle_run(workspace, run_name)
    except (KaggleBackendError, WorkspaceError, ValueError, OSError) as exc:
        code = "REMOTE_JOB_NOT_READY" if "REMOTE_JOB_NOT_READY" in str(exc) else "REMOTE_RECOVERY_FAILED"
        _fail(code, exc, json_output)

    payload = {
        "ok": True,
        "run": record["name"],
        "status": record["status"],
        "mlflow_run_id": record.get("mlflow_run_id"),
        "manifest_hash": record.get("artifact_manifest_hash"),
    }
    if json_output:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
    else:
        typer.echo(f"Recovered {record['name']}: {record['status']}")


@remote_app.command("logs")
def remote_logs(
    run_name: str = typer.Argument(..., help="Kaggle-backed Run ID."),
) -> None:
    """Print Kaggle provider logs for a remote Run."""
    try:
        workspace = discover_workspace()
        typer.echo(kaggle_remote_logs(workspace, run_name), nl=False)
    except (KaggleBackendError, WorkspaceError, ValueError, OSError) as exc:
        _fail("REMOTE_LOGS_FAILED", exc, False)
