from __future__ import annotations

import json
from pathlib import Path

import typer

from .submissions import (
    KaggleSubmissionProvider,
    SubmissionError,
    budget_status,
    create_submission,
    list_submission_summaries,
    send_submission,
    sync_submission,
    validate_submission,
)
from .workspace import WorkspaceError, discover_workspace


submit_app = typer.Typer(
    no_args_is_help=True,
    help="Create, validate, send, and synchronize competition submissions.",
)


def _fail(exc: Exception, json_output: bool) -> None:
    message = str(exc)
    code = message.split(":", 1)[0] if ":" in message else "SUBMISSION_FAILED"
    if json_output:
        typer.echo(json.dumps({"ok": False, "error": {"code": code, "message": message}}))
    else:
        typer.echo(f"ArenaPilot failed: {message}", err=True)
    raise typer.Exit(code=1)


@submit_app.command("create")
def submission_create_command(
    run: str = typer.Option(..., "--run", help="VERIFIED source Run, e.g. run001."),
    file: Path | None = typer.Option(
        None,
        "--file",
        help="Provider-ready CSV. Defaults to outputs/runs/<run>/submission.csv.",
    ),
    message: str | None = typer.Option(None, "--message", help="Optional message saved for send."),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    """Copy a provider-ready CSV into an immutable Arena submission artifact."""
    try:
        workspace = discover_workspace()
        record = create_submission(workspace, run_name=run, file_path=file, message=message)
    except (SubmissionError, WorkspaceError, ValueError, OSError) as exc:
        _fail(exc, json_output)

    payload = {
        "ok": True,
        "submission": record["name"],
        "status": record["status"],
        "run": record["source_run_name"],
        "experiment": record["experiment_name"],
        "file": record["file_path"],
        "sha256": record["file_sha256"],
    }
    if json_output:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
    else:
        typer.echo(
            f"Created {record['name']} from {record['source_run_name']} "
            f"({str(record['file_sha256'])[:12]}...)"
        )


@submit_app.command("validate")
def submission_validate_command(
    name: str = typer.Argument(..., help="Submission ID, e.g. sub001."),
    sample: Path | None = typer.Option(
        None,
        "--sample",
        help="Sample submission CSV. Defaults to data/raw/sample_submission.csv.",
    ),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    """Validate schema, row/ID alignment, and prediction semantics."""
    try:
        workspace = discover_workspace()
        record = validate_submission(workspace, name, sample_path=sample)
    except (SubmissionError, WorkspaceError, ValueError, OSError) as exc:
        _fail(exc, json_output)

    payload = {"ok": True, "submission": name, "status": record["status"]}
    if json_output:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
    else:
        typer.echo(f"Validated {name}.")


@submit_app.command("send")
def submission_send_command(
    name: str = typer.Argument(..., help="Validated submission ID, e.g. sub001."),
    message: str | None = typer.Option(None, "--message", help="Kaggle submission message."),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    """Enforce the local submission budget and send to Kaggle."""
    try:
        workspace = discover_workspace()
        record = send_submission(workspace, name, message=message)
    except (SubmissionError, WorkspaceError, ValueError, OSError) as exc:
        _fail(exc, json_output)

    payload = {
        "ok": True,
        "submission": name,
        "status": record["status"],
        "platform_submission_id": record["platform_submission_id"],
        "message": record["message"],
    }
    if json_output:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
    else:
        typer.echo(
            f"Submitted {name} to Kaggle: ref={record['platform_submission_id']}"
        )


@submit_app.command("status")
def submission_status_command(
    name: str = typer.Argument(..., help="Submission ID, e.g. sub001."),
    sync: bool = typer.Option(True, "--sync/--no-sync", help="Synchronize Kaggle status first."),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    """Show local state and optionally synchronize the Kaggle score."""
    try:
        workspace = discover_workspace()
        if sync:
            record = sync_submission(workspace, name)
        else:
            summaries = list_submission_summaries(workspace)
            match = next((item for item in summaries if item["id"] == name), None)
            if match is None:
                raise SubmissionError(f"SUBMISSION_NOT_FOUND: {name}")
            record = match
    except (SubmissionError, WorkspaceError, ValueError, OSError) as exc:
        _fail(exc, json_output)

    if "name" in record:
        payload = {
            "ok": True,
            "submission": record["name"],
            "status": record["status"],
            "platform_submission_id": record["platform_submission_id"],
            "public_score": record["public_score"],
            "private_score": record["private_score"],
        }
    else:
        payload = {"ok": True, **record}
    if json_output:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
    else:
        typer.echo(f"Submission: {name}")
        typer.echo(f"Status: {payload['status']}")
        typer.echo(f"Kaggle ref: {payload.get('platform_submission_id') or '-'}")
        typer.echo(f"Public score: {payload.get('public_score') if payload.get('public_score') is not None else '-'}")
        typer.echo(f"Private score: {payload.get('private_score') if payload.get('private_score') is not None else '-'}")


@submit_app.command("budget")
def submission_budget_command(
    provider: bool = typer.Option(False, "--provider", help="Also query Kaggle submission limits."),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    """Show ArenaPilot submission budget usage and optional Kaggle limits."""
    try:
        workspace = discover_workspace()
        payload = {
            "ok": True,
            **budget_status(
                workspace,
                provider=KaggleSubmissionProvider() if provider else None,
            ),
        }
    except (SubmissionError, WorkspaceError, ValueError, OSError) as exc:
        _fail(exc, json_output)

    if json_output:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
    else:
        daily_remaining = payload["daily_remaining"]
        total_remaining = payload["total_remaining"]
        typer.echo(
            f"Daily: {payload['daily_used']}/{payload['daily_budget'] or 'unlimited'} "
            f"(remaining={daily_remaining if daily_remaining is not None else 'unlimited'})"
        )
        typer.echo(
            f"Total: {payload['total_used']}/{payload['total_budget'] or 'unlimited'} "
            f"(remaining={total_remaining if total_remaining is not None else 'unlimited'})"
        )
        if payload.get("provider_limits") is not None:
            typer.echo("Kaggle limits: " + json.dumps(payload["provider_limits"], sort_keys=True))


def submissions_command(
    sync: bool = typer.Option(False, "--sync", help="Synchronize submitted Kaggle scores first."),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    """List ArenaPilot submissions and their known scores."""
    try:
        workspace = discover_workspace()
        rows = list_submission_summaries(workspace, sync=sync)
    except (SubmissionError, WorkspaceError, ValueError, OSError) as exc:
        _fail(exc, json_output)

    if json_output:
        typer.echo(json.dumps({"ok": True, "submissions": rows}, indent=2, sort_keys=True))
    else:
        if not rows:
            typer.echo("No submissions.")
            return
        for row in rows:
            public = row["public_score"] if row["public_score"] is not None else "-"
            typer.echo(
                f"{row['id']}  {row['status']}  {row['run']}  "
                f"public={public}  {row['message'] or ''}"
            )
