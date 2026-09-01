from __future__ import annotations

from . import cli as _cli
from .agent_cli import contract_command, experiment_configure_command
from .execution import run_experiment
from .memory_cli import (
    evidence_app,
    failure_app,
    finding_app,
    fingerprint_app,
    knowledge_app,
    learn_command,
)
from .promotion_cli import independence_app, technique_app
from .remote_cli import remote_app
from .submit_cli import submit_app, submissions_command

# Keep the existing command surface stable while routing experiment execution
# through the backend dispatcher. The original CLI command resolves this global
# at call time, so the public entry point can upgrade execution without
# duplicating the full command definition.
_cli.run_local_experiment = run_experiment
_cli.experiment_app.command("configure")(experiment_configure_command)
_cli.app.command("contract")(contract_command)
_cli.app.add_typer(remote_app, name="remote")
_cli.app.add_typer(submit_app, name="submit")
_cli.app.command("submissions")(submissions_command)
_cli.app.add_typer(fingerprint_app, name="fingerprint")
_cli.app.add_typer(evidence_app, name="evidence")
_cli.app.add_typer(finding_app, name="finding")
_cli.app.add_typer(knowledge_app, name="knowledge")
_cli.app.add_typer(failure_app, name="failure")
_cli.app.add_typer(independence_app, name="independence")
_cli.app.add_typer(technique_app, name="technique")
_cli.app.command("learn")(learn_command)

app = _cli.app
