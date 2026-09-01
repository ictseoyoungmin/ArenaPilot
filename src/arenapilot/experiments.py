from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .db import (
    create_experiment_record,
    experiment_parents,
    freeze_experiment_record,
    get_experiment,
    list_experiments,
    next_experiment_name,
)
from .models import ExperimentParent, ExperimentSpec
from .runstore import get_canonical_run_for_experiment
from .tracking import mlflow_run_summary
from .validation import validation_hashes
from .workspace import (
    Workspace,
    WorkspaceError,
    load_arena_config,
    load_experiment_spec,
    load_validation_spec,
    save_experiment_spec,
    save_frozen_experiment_spec,
)


class ExperimentError(WorkspaceError):
    pass


def _hash_spec(spec: ExperimentSpec) -> str:
    raw = json.dumps(
        spec.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def create_experiment(
    workspace: Workspace,
    *,
    title: str,
    hypothesis: str,
    model_family: str,
    parent: str | None = None,
    relation: str = "derived_from",
    backend: str | None = None,
) -> ExperimentSpec:
    config = load_arena_config(workspace)
    if config.competition.status != "ready" or config.validation.active is None:
        raise ExperimentError("competition must be ready with an active validation")

    validation = load_validation_spec(workspace, config.validation.active)
    if validation.status != "active":
        raise ExperimentError(f"active validation is not active on disk: {validation.id}")

    parents: list[ExperimentParent] = []
    if parent is not None:
        parent_row = get_experiment(workspace.db_path, workspace.competition_id, parent)
        if parent_row is None:
            raise ExperimentError(f"parent experiment not found: {parent}")
        if parent_row["status"] == "draft":
            raise ExperimentError("parent experiment must be frozen before derivation")
        parents.append(ExperimentParent(experiment=parent, relation=relation))

    name = next_experiment_name(workspace.db_path, workspace.competition_id)
    comparison_hash, _ = validation_hashes(config, validation)
    spec = ExperimentSpec.model_validate(
        {
            "schema_version": 1,
            "id": name,
            "title": title,
            "parents": [item.model_dump(mode="json") for item in parents],
            "hypothesis": hypothesis,
            "validation": validation.id,
            "pipeline": {},
            "model": {"family": model_family, "params": {}},
            "seed": {"policy": "fixed", "value": config.workspace.default_seed},
            "runtime": {
                "backend": backend or config.compute.default_backend,
                "resources": {},
            },
            "tags": [],
            "notes": None,
        }
    )

    save_experiment_spec(workspace, spec)
    try:
        create_experiment_record(
            workspace.db_path,
            workspace.competition_id,
            spec,
            comparison_hash,
            workspace.experiment_path(name),
        )
    except Exception:
        workspace.experiment_path(name).unlink(missing_ok=True)
        raise
    return spec


def configure_experiment(
    workspace: Workspace,
    name: str,
    *,
    model_params: dict[str, object] | None = None,
    pipeline: dict[str, object] | None = None,
    seed: int | None = None,
    backend: str | None = None,
    resources: dict[str, object] | None = None,
    tags: list[str] | None = None,
) -> ExperimentSpec:
    """Update agent-authorable fields of a draft Experiment through the runtime boundary."""
    row = get_experiment(workspace.db_path, workspace.competition_id, name)
    if row is None:
        raise ExperimentError(f"experiment not found: {name}")
    if row["status"] != "draft":
        raise ExperimentError(f"experiment configuration is immutable after freeze: {name}")

    spec = load_experiment_spec(workspace, name)
    if spec.id != name:
        raise ExperimentError(f"experiment spec id mismatch: expected {name}, got {spec.id}")
    if spec.validation != row["validation_id"]:
        raise ExperimentError("experiment validation binding cannot be changed after creation")
    _validate_lineage_matches_db(workspace, row, spec)

    payload = spec.model_dump(mode="json")
    if model_params is not None:
        model = dict(payload["model"])
        model["params"] = model_params
        payload["model"] = model
    if pipeline is not None:
        payload["pipeline"] = pipeline
    if seed is not None:
        payload["seed"] = {"policy": "fixed", "value": seed}
    if backend is not None:
        runtime = dict(payload["runtime"])
        runtime["backend"] = backend
        payload["runtime"] = runtime
    if resources is not None:
        runtime = dict(payload["runtime"])
        runtime["resources"] = resources
        payload["runtime"] = runtime
    if tags is not None:
        payload["tags"] = tags

    updated = ExperimentSpec.model_validate(payload)
    save_experiment_spec(workspace, updated)
    return updated


def _validate_lineage_matches_db(workspace: Workspace, row: dict[str, object], spec: ExperimentSpec) -> None:
    persisted = {
        (str(parent["name"]), str(parent["relation"]))
        for parent in experiment_parents(workspace.db_path, str(row["id"]))
    }
    declared = {(parent.experiment, parent.relation) for parent in spec.parents}
    if declared != persisted:
        raise ExperimentError("experiment parent lineage differs from the persisted draft lineage")


def freeze_experiment(workspace: Workspace, name: str) -> tuple[dict[str, object], Path]:
    row = get_experiment(workspace.db_path, workspace.competition_id, name)
    if row is None:
        raise ExperimentError(f"experiment not found: {name}")

    spec = load_experiment_spec(workspace, name)
    if spec.id != name:
        raise ExperimentError(f"experiment spec id mismatch: expected {name}, got {spec.id}")
    if spec.validation != row["validation_id"]:
        raise ExperimentError("experiment validation binding cannot be changed after creation")
    _validate_lineage_matches_db(workspace, row, spec)

    config_hash = _hash_spec(spec)
    if row["status"] == "frozen":
        if row["config_hash"] != config_hash:
            raise ExperimentError("FROZEN_SPEC_MODIFIED")
        snapshot = workspace.frozen_experiment_path(name, config_hash)
        if not snapshot.is_file():
            snapshot = save_frozen_experiment_spec(workspace, spec, config_hash)
        return row, snapshot
    if row["status"] != "draft":
        raise ExperimentError(f"experiment cannot be frozen from status {row['status']}")

    snapshot = save_frozen_experiment_spec(workspace, spec, config_hash)
    try:
        frozen = freeze_experiment_record(
            workspace.db_path,
            workspace.competition_id,
            name,
            config_hash,
        )
    except Exception:
        snapshot.unlink(missing_ok=True)
        raise
    return frozen, snapshot


def show_experiment(workspace: Workspace, name: str) -> dict[str, object]:
    row = get_experiment(workspace.db_path, workspace.competition_id, name)
    if row is None:
        raise ExperimentError(f"experiment not found: {name}")
    spec = load_experiment_spec(workspace, name)
    integrity: bool | None = None
    if row["status"] == "frozen" and row["config_hash"]:
        integrity = _hash_spec(spec) == row["config_hash"]

    config = load_arena_config(workspace)
    metric_name = config.metric.name if config.metric else None
    canonical = get_canonical_run_for_experiment(
        workspace.db_path,
        workspace.competition_id,
        name,
    )
    tracking = mlflow_run_summary(
        str(canonical["mlflow_run_id"]) if canonical and canonical.get("mlflow_run_id") else None,
        metric_name,
    )
    return {
        "record": row,
        "spec": spec.model_dump(mode="json"),
        "parents": experiment_parents(workspace.db_path, str(row["id"])),
        "spec_integrity": integrity,
        "canonical_run": canonical,
        "tracking": tracking,
    }


def list_experiment_summaries(workspace: Workspace) -> list[dict[str, object]]:
    rows = list_experiments(workspace.db_path, workspace.competition_id)
    config = load_arena_config(workspace)
    metric_name = config.metric.name if config.metric else None
    summaries: list[dict[str, object]] = []
    for row in rows:
        integrity: bool | None = None
        if row["status"] == "frozen" and row["config_hash"]:
            try:
                integrity = _hash_spec(load_experiment_spec(workspace, str(row["name"]))) == row["config_hash"]
            except Exception:
                integrity = False
        canonical = get_canonical_run_for_experiment(
            workspace.db_path,
            workspace.competition_id,
            str(row["name"]),
        )
        tracking = mlflow_run_summary(
            str(canonical["mlflow_run_id"]) if canonical and canonical.get("mlflow_run_id") else None,
            metric_name,
        )
        summaries.append(
            {
                "id": row["name"],
                "title": row["title"],
                "status": row["status"],
                "validation": row["validation_id"],
                "config_hash": row["config_hash"],
                "spec_integrity": integrity,
                "canonical_run": canonical["name"] if canonical else None,
                "tracked": tracking["tracked"],
                "primary_metric": tracking["primary_metric"],
            }
        )
    return summaries


def experiment_lineage(workspace: Workspace, name: str) -> list[dict[str, str]]:
    row = get_experiment(workspace.db_path, workspace.competition_id, name)
    if row is None:
        raise ExperimentError(f"experiment not found: {name}")

    edges: list[dict[str, str]] = []
    visited: set[str] = set()

    def walk(child_row: dict[str, object]) -> None:
        child_name = str(child_row["name"])
        if child_name in visited:
            return
        visited.add(child_name)
        for parent in experiment_parents(workspace.db_path, str(child_row["id"])):
            parent_name = str(parent["name"])
            edges.append(
                {
                    "parent": parent_name,
                    "child": child_name,
                    "relation": str(parent["relation"]),
                }
            )
            parent_row = get_experiment(workspace.db_path, workspace.competition_id, parent_name)
            if parent_row is not None:
                walk(parent_row)

    walk(row)
    return edges