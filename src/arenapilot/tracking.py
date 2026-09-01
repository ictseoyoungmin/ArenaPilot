from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

from mlflow.tracking import MlflowClient

from .models import ArenaConfig, ExperimentSpec
from .workspace import Workspace, WorkspaceError, load_arena_config


class TrackingError(WorkspaceError):
    pass


def arenapilot_home() -> Path:
    configured = os.environ.get("ARENAPILOT_HOME")
    path = Path(configured).expanduser() if configured else Path.home() / ".arenapilot"
    path.mkdir(parents=True, exist_ok=True)
    return path.resolve()


def tracking_uri() -> str:
    path = arenapilot_home() / "mlflow.db"
    return f"sqlite:///{path.as_posix()}"


def artifact_root(config: ArenaConfig) -> Path:
    root = (
        arenapilot_home()
        / "mlartifacts"
        / config.competition.platform
        / config.competition.slug
    )
    root.mkdir(parents=True, exist_ok=True)
    return root.resolve()


def mlflow_experiment_name(config: ArenaConfig) -> str:
    return (
        f"arena/{config.competition.platform}/"
        f"{config.tracking.experiment_name}"
    )


def _client() -> MlflowClient:
    try:
        return MlflowClient(tracking_uri=tracking_uri())
    except Exception as exc:
        raise TrackingError(f"MLFLOW_CLIENT_INIT_FAILED: {exc}") from exc


def _ensure_experiment(client: MlflowClient, config: ArenaConfig) -> str:
    name = mlflow_experiment_name(config)
    try:
        existing = client.get_experiment_by_name(name)
        if existing is not None:
            return existing.experiment_id
        return client.create_experiment(
            name,
            artifact_location=artifact_root(config).as_uri(),
        )
    except Exception as exc:
        raise TrackingError(f"MLFLOW_EXPERIMENT_FAILED: {exc}") from exc


def _param_value(value: Any) -> str:
    if isinstance(value, (str, int, float, bool)) and value is not None:
        return str(value)
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


def _flatten_params(value: Any, prefix: str = "") -> dict[str, str]:
    result: dict[str, str] = {}
    if isinstance(value, dict):
        for key, item in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(item, dict):
                result.update(_flatten_params(item, name))
            else:
                result[name] = _param_value(item)
        return result
    if prefix:
        result[prefix] = _param_value(value)
    return result


def _numeric_metrics(payload: object) -> dict[str, float]:
    if not isinstance(payload, dict):
        return {}
    metrics: dict[str, float] = {}
    for key, value in payload.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            number = float(value)
            if math.isfinite(number):
                metrics[str(key)] = number
    return metrics


def _fold_metrics(payload: object) -> dict[str, float]:
    if not isinstance(payload, dict) or not isinstance(payload.get("folds"), list):
        return {}
    result: dict[str, float] = {}
    for index, fold in enumerate(payload["folds"]):
        if not isinstance(fold, dict):
            continue
        fold_number = fold.get("fold", index)
        for key, value in fold.items():
            if key == "fold":
                continue
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                number = float(value)
                if math.isfinite(number):
                    result[f"fold_{fold_number}_{key}"] = number
    return result


def _read_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TrackingError(f"cannot read tracking artifact {path.name}") from exc


def _existing_mlflow_run(
    client: MlflowClient,
    experiment_id: str,
    internal_run_id: str,
):
    try:
        matches = client.search_runs(
            experiment_ids=[experiment_id],
            filter_string=f"tags.arena.run_internal_id = '{internal_run_id}'",
            max_results=2,
        )
    except Exception as exc:
        raise TrackingError(f"MLFLOW_RUN_LOOKUP_FAILED: {exc}") from exc
    if len(matches) > 1:
        raise TrackingError("multiple MLflow runs found for one ArenaPilot run")
    return matches[0] if matches else None


def ingest_verified_run(
    workspace: Workspace,
    run_row: dict[str, object],
    run_dir: Path,
    experiment_spec: ExperimentSpec,
) -> str:
    config = load_arena_config(workspace)
    client = _client()
    experiment_id = _ensure_experiment(client, config)
    internal_run_id = str(run_row["id"])
    existing = _existing_mlflow_run(client, experiment_id, internal_run_id)

    tags = {
        "arena.competition_id": workspace.competition_id,
        "arena.competition_slug": config.competition.slug,
        "arena.experiment_id": str(run_row["experiment_id"]),
        "arena.experiment_name": str(run_row["experiment_name"]),
        "arena.run_internal_id": internal_run_id,
        "arena.run_name": str(run_row["name"]),
        "arena.validation_id": experiment_spec.validation,
        "arena.backend": str(run_row["backend"]),
        "arena.spec_hash": str(run_row["spec_hash"]),
        "arena.hypothesis": experiment_spec.hypothesis,
    }

    try:
        if existing is None:
            created = client.create_run(
                experiment_id=experiment_id,
                tags=tags,
                run_name=f"{experiment_spec.id}/{run_row['name']}",
            )
            mlflow_run_id = created.info.run_id
        else:
            mlflow_run_id = existing.info.run_id
            for key, value in tags.items():
                client.set_tag(mlflow_run_id, key, value)

        params = _flatten_params(
            {
                "model": experiment_spec.model,
                "pipeline": experiment_spec.pipeline,
                "seed": experiment_spec.seed.model_dump(mode="json"),
                "runtime": experiment_spec.runtime.model_dump(mode="json"),
            }
        )
        for key, value in params.items():
            client.log_param(mlflow_run_id, key, value)

        metrics = _numeric_metrics(_read_json(run_dir / "metrics.json"))
        fold_metrics_path = run_dir / "fold_metrics.json"
        if fold_metrics_path.is_file():
            metrics.update(_fold_metrics(_read_json(fold_metrics_path)))
        result = _read_json(run_dir / "result.json")
        if isinstance(result, dict):
            duration = result.get("duration_seconds")
            if isinstance(duration, (int, float)) and not isinstance(duration, bool):
                number = float(duration)
                if math.isfinite(number):
                    metrics["duration_seconds"] = number

        for key, value in metrics.items():
            client.log_metric(mlflow_run_id, key, value)

        client.log_artifacts(mlflow_run_id, str(run_dir))
        client.set_terminated(mlflow_run_id, status="FINISHED")
        return mlflow_run_id
    except TrackingError:
        raise
    except Exception as exc:
        if existing is None and "mlflow_run_id" in locals():
            try:
                client.set_terminated(mlflow_run_id, status="FAILED")
            except Exception:
                pass
        raise TrackingError(f"MLFLOW_INGEST_FAILED: {exc}") from exc


def mlflow_run_summary(mlflow_run_id: str | None, metric_name: str | None = None) -> dict[str, object]:
    if not mlflow_run_id:
        return {
            "tracked": False,
            "mlflow_run_id": None,
            "primary_metric": None,
        }
    client = _client()
    try:
        run = client.get_run(mlflow_run_id)
    except Exception as exc:
        raise TrackingError(f"MLFLOW_RUN_READ_FAILED: {exc}") from exc
    metric = run.data.metrics.get(metric_name) if metric_name else None
    return {
        "tracked": True,
        "mlflow_run_id": mlflow_run_id,
        "mlflow_status": run.info.status,
        "primary_metric": metric,
        "metrics": dict(run.data.metrics),
    }
