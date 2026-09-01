from __future__ import annotations

from . import cli as _cli
from .execution import run_experiment
from .remote_cli import remote_app
from .submit_cli import submit_app, submissions_command

# Keep the existing command surface stable while routing experiment execution
# through the backend dispatcher. The original CLI command resolves this global
# at call time, so the public entry point can upgrade execution without
# duplicating the full command definition.
_cli.run_local_experiment = run_experiment
_cli.app.add_typer(remote_app, name="remote")
_cli.app.add_typer(submit_app, name="submit")
_cli.app.command("submissions")(submissions_command)

app = _cli.app
