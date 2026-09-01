from __future__ import annotations

import json
from pathlib import Path

import typer

from . import __version__
from .db import initialize_database, read_schema_version
from .experiments import (
    ExperimentError,
    create_experiment,
    experiment_lineage,
    freeze_experiment,
    list_experiment_summaries,
    show_experiment,
)
from .intake import configure_intake
from .models import MetricDirection, PredictionType, SplitType, TaskType
from .runs import (
    RunError,
    list_run_summaries,
    run_local_experiment,
    run_logs,
    show_run,
    verify_run,
)
from .validation import activate_validation, configure_validation
from .workspace import (
    WorkspaceError,
    WorkspaceNotFoundError,
    create_workspace,
    discover_workspace,
    load_arena_config,
)

app = typer.Typer(no_args_is_help=True, help="ArenaPilot competition runtime")
intake_app = typer.Typer(no_args_is_help=True, help="Configure competition intake.")
validation_app = typer.Typer(
    no_args_is_help=True,
    help="Configure and activate validation contracts.",
)
experiment_app = typer.Typer(
    no_args_is_help=True,
    help="Create, freeze, inspect, trace, and run experiments.",
)
run_app = typer.Typer(
    no_args_is_help=True,
    help="Inspect and verify execution runs.",
)
app.add_typer(intake_app, name="intake")
app.add_typer(validation_app, name="validation")
app.add_typer(experiment_app, name="exp")
app.add_typer(run_app, name="run")


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
        _fail("WORKSPACE_INIT_FAILED", exc, json_output)

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
        _fail("WORKSPACE_STATUS_FAILED", exc, json_output)

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


@intake_app.command("set")
def intake_set(
    task: TaskType = typer.Option(..., "--task", help="Competition task type."),
    target: str = typer.Option(..., "--target", help="Target column."),
    metric: str = typer.Option(..., "--metric", help="Primary competition metric."),
    direction: MetricDirection = typer.Option(..., "--direction", help="Metric optimization direction."),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    """Set the competition task, target, and primary metric while still draft."""
    try:
        workspace = discover_workspace()
        config = configure_intake(
            workspace,
            task_type=task,
            target=target,
            metric_name=metric,
            metric_direction=direction,
        )
    except (WorkspaceError, ValueError, OSError) as exc:
        _fail("INTAKE_CONFIG_FAILED", exc, json_output)

    assert config.task is not None
    assert config.metric is not None
    payload = {
        "ok": True,
        "competition_status": config.competition.status,
        "task": config.task.type,
        "target": config.task.target,
        "metric": config.metric.name,
        "direction": config.metric.direction,
    }
    if json_output:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
    else:
        typer.echo(
            f"Configured intake: {config.task.type} target={config.task.target}, "
            f"metric={config.metric.name} ({config.metric.direction})"
        )


@validation_app.command("configure")
def validation_configure(
    name: str = typer.Argument(..., help="Validation ID, e.g. val-v1."),
    split: SplitType = typer.Option(..., "--split", help="Split strategy."),
    prediction: PredictionType = typer.Option(..., "--prediction", help="Prediction semantics."),
    n_splits: int = typer.Option(5, "--n-splits", min=2),
    shuffle: bool = typer.Option(True, "--shuffle/--no-shuffle"),
    random_state: int | None = typer.Option(42, "--random-state"),
    group_column: str | None = typer.Option(None, "--group-column"),
    time_column: str | None = typer.Option(None, "--time-column"),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    """Configure a draft validation using the competition metric."""
    try:
        workspace = discover_workspace()
        spec = configure_validation(
            workspace,
            name,
            split_type=split,
            prediction_type=prediction,
            n_splits=n_splits,
            shuffle=shuffle,
            random_state=random_state,
            group_column=group_column,
            time_column=time_column,
        )
    except (WorkspaceError, ValueError, OSError) as exc:
        _fail("VALIDATION_CONFIG_FAILED", exc, json_output)

    assert spec.prediction is not None
    assert spec.metric is not None
    payload = {
        "ok": True,
        "validation": spec.id,
        "status": spec.status,
        "split": spec.split.type,
        "prediction": spec.prediction.type,
        "metric": spec.metric.name,
    }
    if json_output:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
    else:
        typer.echo(
            f"Configured {spec.id}: {spec.split.type}, "
            f"prediction={spec.prediction.type}, metric={spec.metric.name}"
        )


@validation_app.command("activate")
def validation_activate(
    name: str = typer.Argument(..., help="Validation ID, e.g. val-v1."),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    """Activate a complete validation and move the competition to ready."""
    try:
        workspace = discover_workspace()
        config, spec = activate_validation(workspace, name)
    except (WorkspaceError, ValueError, OSError) as exc:
        _fail("VALIDATION_ACTIVATION_FAILED", exc, json_output)

    payload = {
        "ok": True,
        "validation": spec.id,
        "validation_status": spec.status,
        "competition_status": config.competition.status,
        "active_validation": config.validation.active,
    }
    if json_output:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
    else:
        typer.echo(f"Activated {spec.id}. Competition status: {config.competition.status}.")


@experiment_app.command("new")
def experiment_new(
    title: str = typer.Option(..., "--title", help="Short experiment title."),
    hypothesis: str = typer.Option(..., "--hypothesis", help="Testable experiment hypothesis."),
    model_family: str = typer.Option(..., "--model-family", help="Model family, e.g. catboost."),
    parent: str | None = typer.Option(None, "--from", help="Frozen parent experiment."),
    relation: str = typer.Option("derived_from", "--relation", help="Parent relation."),
    backend: str | None = typer.Option(None, "--backend", help="Optional local/kaggle backend override."),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    """Create a draft experiment bound to the active validation."""
    try:
        workspace = discover_workspace()
        spec = create_experiment(
            workspace,
            title=title,
            hypothesis=hypothesis,
            model_family=model_family,
            parent=parent,
            relation=relation,
            backend=backend,
        )
    except (ExperimentError, WorkspaceError, ValueError, OSError) as exc:
        _fail("EXPERIMENT_CREATE_FAILED", exc, json_output)

    payload = {
        "ok": True,
        "experiment": spec.id,
        "status": "draft",
        "validation": spec.validation,
        "path": str(workspace.experiment_path(spec.id)),
    }
    if json_output:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
    else:
        typer.echo(f"Created {spec.id}: {spec.title} (validation={spec.validation})")


@experiment_app.command("freeze")
def experiment_freeze(
    name: str = typer.Argument(..., help="Experiment ID, e.g. exp001."),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    """Freeze an experiment config and persist an immutable snapshot."""
    try:
        workspace = discover_workspace()
        record, snapshot = freeze_experiment(workspace, name)
    except (ExperimentError, WorkspaceError, ValueError, OSError) as exc:
        code = "FROZEN_SPEC_MODIFIED" if str(exc) == "FROZEN_SPEC_MODIFIED" else "EXPERIMENT_FREEZE_FAILED"
        _fail(code, exc, json_output)

    payload = {
        "ok": True,
        "experiment": name,
        "status": record["status"],
        "config_hash": record["config_hash"],
        "snapshot": str(snapshot),
    }
    if json_output:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
    else:
        typer.echo(f"Frozen {name}: {record['config_hash']}")


@experiment_app.command("run")
def experiment_run(
    name: str = typer.Argument(..., help="Frozen experiment ID, e.g. exp001."),
    backend: str | None = typer.Option(None, "--backend", help="Backend override. This slice supports local only."),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    """Execute a frozen experiment as a new run."""
    try:
        workspace = discover_workspace()
        record = run_local_experiment(workspace, name, backend=backend)
    except (RunError, ExperimentError, WorkspaceError, ValueError, OSError) as exc:
        if str(exc) == "EXPERIMENT_NOT_FROZEN":
            code = "EXPERIMENT_NOT_FROZEN"
        elif str(exc) == "FROZEN_SPEC_MODIFIED":
            code = "FROZEN_SPEC_MODIFIED"
        else:
            code = "RUN_EXECUTION_FAILED"
        _fail(code, exc, json_output)

    payload = {
        "ok": True,
        "run": record["name"],
        "experiment": record["experiment_name"],
        "status": record["status"],
        "backend": record["backend"],
        "manifest_hash": record["artifact_manifest_hash"],
    }
    if json_output:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
    else:
        typer.echo(f"{record['name']} {record['status']} ({record['experiment_name']}, {record['backend']})")


@experiment_app.command("show")
def experiment_show(
    name: str = typer.Argument(..., help="Experiment ID, e.g. exp001."),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    """Show an experiment spec, state, and lineage parents."""
    try:
        workspace = discover_workspace()
        payload = {"ok": True, **show_experiment(workspace, name)}
    except (ExperimentError, WorkspaceError, ValueError, OSError) as exc:
        _fail("EXPERIMENT_SHOW_FAILED", exc, json_output)

    if json_output:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
    else:
        record = payload["record"]
        typer.echo(f"Experiment: {record['name']} — {record['title']}")
        typer.echo(f"Status: {record['status']}")
        typer.echo(f"Validation: {record['validation_id']}")
        typer.echo(f"Spec integrity: {payload['spec_integrity']}")


@experiment_app.command("list")
def experiment_list(
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    """List experiments in creation order."""
    try:
        workspace = discover_workspace()
        experiments = list_experiment_summaries(workspace)
    except (ExperimentError, WorkspaceError, ValueError, OSError) as exc:
        _fail("EXPERIMENT_LIST_FAILED", exc, json_output)

    if json_output:
        typer.echo(json.dumps({"ok": True, "experiments": experiments}, indent=2, sort_keys=True))
    else:
        if not experiments:
            typer.echo("No experiments.")
            return
        for item in experiments:
            typer.echo(f"{item['id']}  {item['status']}  {item['validation']}  {item['title']}")


@experiment_app.command("lineage")
def experiment_lineage_command(
    name: str = typer.Argument(..., help="Experiment ID, e.g. exp002."),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    """Show parent edges from an experiment back through its ancestry."""
    try:
        workspace = discover_workspace()
        edges = experiment_lineage(workspace, name)
    except (ExperimentError, WorkspaceError, ValueError, OSError) as exc:
        _fail("EXPERIMENT_LINEAGE_FAILED", exc, json_output)

    if json_output:
        typer.echo(json.dumps({"ok": True, "experiment": name, "edges": edges}, indent=2, sort_keys=True))
    else:
        if not edges:
            typer.echo(f"{name} has no parents.")
            return
        for edge in edges:
            typer.echo(f"{edge['parent']} --{edge['relation']}--> {edge['child']}")


@run_app.command("show")
def run_show_command(
    name: str = typer.Argument(..., help="Run ID, e.g. run001."),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    """Show a run record and artifact manifest."""
    try:
        workspace = discover_workspace()
        payload = {"ok": True, **show_run(workspace, name)}
    except (RunError, WorkspaceError, ValueError, OSError) as exc:
        _fail("RUN_SHOW_FAILED", exc, json_output)

    if json_output:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
    else:
        record = payload["record"]
        typer.echo(f"Run: {record['name']} — {record['experiment_name']}")
        typer.echo(f"Status: {record['status']}")
        typer.echo(f"Backend: {record['backend']}")
        typer.echo(f"Exit code: {record['exit_code']}")


@run_app.command("list")
def run_list_command(
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    """List execution runs in creation order."""
    try:
        workspace = discover_workspace()
        runs = list_run_summaries(workspace)
    except (RunError, WorkspaceError, ValueError, OSError) as exc:
        _fail("RUN_LIST_FAILED", exc, json_output)

    if json_output:
        typer.echo(json.dumps({"ok": True, "runs": runs}, indent=2, sort_keys=True))
    else:
        if not runs:
            typer.echo("No runs.")
            return
        for item in runs:
            typer.echo(f"{item['id']}  {item['status']}  {item['backend']}  {item['experiment']}")


@run_app.command("logs")
def run_logs_command(
    name: str = typer.Argument(..., help="Run ID, e.g. run001."),
) -> None:
    """Print captured stdout/stderr for a run."""
    try:
        workspace = discover_workspace()
        typer.echo(run_logs(workspace, name), nl=False)
    except (RunError, WorkspaceError, ValueError, OSError) as exc:
        _fail("RUN_LOGS_FAILED", exc, False)


@run_app.command("verify")
def run_verify_command(
    name: str = typer.Argument(..., help="Completed or invalid run ID."),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    """Re-verify the standard artifact contract for a run."""
    try:
        workspace = discover_workspace()
        record = verify_run(workspace, name)
    except (RunError, WorkspaceError, ValueError, OSError) as exc:
        _fail("RUN_VERIFICATION_FAILED", exc, json_output)

    payload = {
        "ok": True,
        "run": record["name"],
        "status": record["status"],
        "manifest_hash": record["artifact_manifest_hash"],
    }
    if json_output:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
    else:
        typer.echo(f"Verified {record['name']}: {record['artifact_manifest_hash']}")


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
