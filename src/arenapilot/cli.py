from __future__ import annotations

import json
from pathlib import Path

import typer

from . import __version__
from .db import initialize_database, read_schema_version
from .workspace import (
    WorkspaceError,
    WorkspaceNotFoundError,
    create_workspace,
    discover_workspace,
    load_arena_config,
)

app = typer.Typer(no_args_is_help=True, help="ArenaPilot competition runtime")


@app.command()
def version() -> None:
    """Print the ArenaPilot version."""
    typer.echo(__version__)


@app.command("init")
def init_workspace(
    competition: str = typer.Argument(..., help="Competition reference, e.g. kaggle:titanic."),
    path: Path | None = typer.Option(None, "--path", help="Workspace destination. Defaults to the competition slug."),
    title: str | None = typer.Option(None, "--title", help="Optional human-readable competition title."),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    """Create a draft ArenaPilot competition workspace atomically."""
    try:
        workspace = create_workspace(competition, path, title=title)
    except WorkspaceError as exc:
        if json_output:
            typer.echo(json.dumps({"ok": False, "error": {"code": "WORKSPACE_INIT_FAILED", "message": str(exc)}}))
        else:
            typer.echo(f"ArenaPilot init failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    payload = {
        "ok": True,
        "workspace": str(workspace.root),
        "competition_id": workspace.competition_id,
        "workspace_id": workspace.workspace_id,
        "status": "draft",
    }
    if json_output:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
    else:
        typer.echo(f"Initialized ArenaPilot workspace: {workspace.root}")
        typer.echo("Status: draft — configure competition intake and validation before experiments.")


@app.command()
def status(json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON.")) -> None:
    """Show the current workspace setup status."""
    try:
        workspace = discover_workspace()
        config = load_arena_config(workspace)
    except (WorkspaceError, ValueError, OSError) as exc:
        if json_output:
            typer.echo(json.dumps({"ok": False, "error": {"code": "WORKSPACE_STATUS_FAILED", "message": str(exc)}}))
        else:
            typer.echo(f"ArenaPilot status failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    payload = {
        "ok": True,
        "workspace": str(workspace.root),
        "competition": f"{config.competition.platform}:{config.competition.slug}",
        "competition_status": config.competition.status,
        "task_configured": config.task is not None,
        "metric_configured": config.metric is not None,
        "active_validation": config.validation.active,
    }
    if json_output:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
    else:
        typer.echo(f"Competition: {payload['competition']}")
        typer.echo(f"Status: {config.competition.status}")
        typer.echo(f"Task configured: {payload['task_configured']}")
        typer.echo(f"Metric configured: {payload['metric_configured']}")
        typer.echo(f"Active validation: {config.validation.active or '-'}")


@app.command()
def doctor(json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON.")) -> None:
    """Check the current ArenaPilot workspace and core state."""
    checks: dict[str, object] = {"version": __version__}
    ok = True
    try:
        workspace = discover_workspace()
        checks["workspace"] = str(workspace.root)
        checks["competition_id"] = workspace.competition_id
        config = load_arena_config(workspace)
        checks["config"] = "valid"
        checks["competition_status"] = config.competition.status
        initialize_database(workspace.db_path)
        checks["database"] = str(workspace.db_path)
        checks["database_schema"] = read_schema_version(workspace.db_path)
    except (WorkspaceNotFoundError, ValueError, OSError, RuntimeError) as exc:
        ok = False
        checks["workspace_error"] = str(exc)

    payload = {"ok": ok, "checks": checks}
    if json_output:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
    else:
        typer.echo("ArenaPilot doctor: OK" if ok else "ArenaPilot doctor: FAILED")
        for key, value in checks.items():
            typer.echo(f"- {key}: {value}")

    if not ok:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
