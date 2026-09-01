from __future__ import annotations

import json

import typer

from .agent_contract import contract_payload
from .experiments import ExperimentError, configure_experiment
from .workspace import WorkspaceError, discover_workspace


def _json_object(raw: str | None, label: str) -> dict[str, object] | None:
    if raw is None:
        return None
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ExperimentError(f"{label} must be valid JSON") from exc
    if not isinstance(value, dict):
        raise ExperimentError(f"{label} must be a JSON object")
    return value


def contract_command(
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    """Show the stable Agent-to-Runtime contract implemented by this build."""
    payload = contract_payload()
    if json_output:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
        return
    typer.echo(f"ArenaPilot agent contract v{payload['contract_version']}")
    typer.echo(f"Mutation boundary: {payload['mutation_boundary']}")
    typer.echo("Capabilities:")
    for item in payload["capabilities"]:
        typer.echo(f"- {item}")


def experiment_configure_command(
    name: str = typer.Argument(..., help="Draft Experiment ID, e.g. exp001."),
    model_params_json: str | None = typer.Option(
        None,
        "--model-params-json",
        help="Replacement model params JSON object.",
    ),
    pipeline_json: str | None = typer.Option(
        None,
        "--pipeline-json",
        help="Replacement pipeline JSON object.",
    ),
    seed: int | None = typer.Option(None, "--seed", help="Fixed experiment seed."),
    backend: str | None = typer.Option(None, "--backend", help="local|kaggle"),
    resources_json: str | None = typer.Option(
        None,
        "--resources-json",
        help="Replacement runtime resources JSON object.",
    ),
    tag: list[str] | None = typer.Option(
        None,
        "--tag",
        help="Replacement tag set. Repeat the option for multiple tags.",
    ),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    """Configure agent-authorable fields of a draft Experiment without editing YAML."""
    try:
        workspace = discover_workspace()
        spec = configure_experiment(
            workspace,
            name,
            model_params=_json_object(model_params_json, "model-params-json"),
            pipeline=_json_object(pipeline_json, "pipeline-json"),
            seed=seed,
            backend=backend,
            resources=_json_object(resources_json, "resources-json"),
            tags=tag,
        )
    except (ExperimentError, WorkspaceError, ValueError, OSError) as exc:
        code = "EXPERIMENT_CONFIG_IMMUTABLE" if "immutable after freeze" in str(exc) else "EXPERIMENT_CONFIG_FAILED"
        if json_output:
            typer.echo(json.dumps({"ok": False, "error": {"code": code, "message": str(exc)}}))
        else:
            typer.echo(f"ArenaPilot failed: {exc}", err=True)
        raise typer.Exit(code=1)

    payload = {
        "ok": True,
        "experiment": spec.id,
        "model": spec.model,
        "pipeline": spec.pipeline,
        "seed": spec.seed.model_dump(mode="json"),
        "runtime": spec.runtime.model_dump(mode="json"),
        "tags": spec.tags,
    }
    if json_output:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
    else:
        typer.echo(f"Configured {spec.id} (backend={spec.runtime.backend}, seed={spec.seed.value}).")
