from __future__ import annotations

import json

import typer

from . import __version__
from .db import initialize_database
from .workspace import WorkspaceNotFoundError, discover_workspace

app = typer.Typer(no_args_is_help=True, help="ArenaPilot competition runtime")


@app.command()
def version() -> None:
    """Print the ArenaPilot version."""
    typer.echo(__version__)


@app.command()
def doctor(json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON.")) -> None:
    """Check whether the current directory is inside an ArenaPilot workspace."""
    checks: dict[str, object] = {"version": __version__}
    ok = True
    try:
        workspace = discover_workspace()
        checks["workspace"] = str(workspace.root)
        checks["competition_id"] = workspace.competition_id
        initialize_database(workspace.db_path)
        checks["database"] = str(workspace.db_path)
    except WorkspaceNotFoundError as exc:
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
