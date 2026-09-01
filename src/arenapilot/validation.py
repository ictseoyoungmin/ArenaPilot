from __future__ import annotations

import hashlib
import json

from .db import sync_validation, sync_validation_activation
from .models import (
    ArenaConfig,
    PredictionConfig,
    PredictionType,
    SplitConfig,
    SplitType,
    TaskType,
    ValidationSpec,
)
from .workspace import (
    Workspace,
    WorkspaceError,
    load_arena_config,
    load_validation_spec,
    save_arena_config,
    save_validation_spec,
)


def _hash_payload(payload: object) -> str:
    raw = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def validation_hashes(config: ArenaConfig, spec: ValidationSpec) -> tuple[str, str]:
    if config.task is None or config.metric is None:
        raise WorkspaceError("competition intake is incomplete")
    if spec.metric is None or spec.prediction is None:
        raise WorkspaceError("validation contract is incomplete")

    comparison_domain = {
        "task": config.task.model_dump(mode="json"),
        "metric": spec.metric.model_dump(mode="json"),
        "split": spec.split.model_dump(mode="json"),
        "prediction": spec.prediction.model_dump(mode="json"),
        "oof": spec.oof.model_dump(mode="json"),
    }
    return (
        _hash_payload(comparison_domain),
        _hash_payload(spec.model_dump(mode="json")),
    )


def _validate_prediction_semantics(task_type: TaskType, prediction_type: PredictionType) -> None:
    if task_type == TaskType.REGRESSION and prediction_type != PredictionType.VALUE:
        raise WorkspaceError("regression validation requires prediction type value")
    if task_type != TaskType.REGRESSION and prediction_type == PredictionType.VALUE:
        raise WorkspaceError("classification validation cannot use prediction type value")


def configure_validation(
    workspace: Workspace,
    name: str,
    *,
    split_type: SplitType,
    prediction_type: PredictionType,
    n_splits: int = 5,
    shuffle: bool = True,
    random_state: int | None = 42,
    group_column: str | None = None,
    time_column: str | None = None,
) -> ValidationSpec:
    config = load_arena_config(workspace)
    if config.task is None or config.metric is None:
        raise WorkspaceError("configure competition intake before validation")

    spec = load_validation_spec(workspace, name)
    if spec.status != "draft":
        raise WorkspaceError(f"only draft validation can be configured: {name}")

    _validate_prediction_semantics(config.task.type, prediction_type)

    payload = spec.model_dump(mode="json")
    payload["split"] = SplitConfig(
        type=split_type,
        n_splits=n_splits,
        shuffle=shuffle,
        random_state=random_state,
        group_column=group_column,
        time_column=time_column,
    ).model_dump(mode="json")
    payload["metric"] = config.metric.model_dump(mode="json")
    payload["prediction"] = PredictionConfig(type=prediction_type).model_dump(mode="json")
    updated = ValidationSpec.model_validate(payload)

    save_validation_spec(workspace, updated)
    try:
        comparison_hash, spec_hash = validation_hashes(config, updated)
        sync_validation(
            workspace.db_path,
            workspace.competition_id,
            updated,
            workspace.validation_path(name),
            comparison_hash,
            spec_hash,
        )
    except Exception:
        save_validation_spec(workspace, spec)
        raise

    return updated


def activate_validation(
    workspace: Workspace,
    name: str,
) -> tuple[ArenaConfig, ValidationSpec]:
    config = load_arena_config(workspace)
    spec = load_validation_spec(workspace, name)

    if config.task is None or config.metric is None:
        raise WorkspaceError("configure competition intake before validation activation")
    if config.validation.active and config.validation.active != name:
        raise WorkspaceError(
            f"another validation is already active: {config.validation.active}"
        )
    if spec.status == "active" and config.validation.active == name:
        return config, spec
    if spec.status != "draft":
        raise WorkspaceError(f"validation must be draft before activation: {name}")
    if spec.metric is None or spec.prediction is None:
        raise WorkspaceError("configure validation metric and prediction before activation")
    if spec.metric != config.metric:
        raise WorkspaceError("validation metric must match competition metric")

    _validate_prediction_semantics(config.task.type, spec.prediction.type)

    spec_payload = spec.model_dump(mode="json")
    spec_payload["status"] = "active"
    active_spec = ValidationSpec.model_validate(spec_payload)

    config_payload = config.model_dump(mode="json")
    config_payload["competition"]["status"] = "ready"
    config_payload["validation"]["active"] = name
    ready_config = ArenaConfig.model_validate(config_payload)

    try:
        save_validation_spec(workspace, active_spec)
        save_arena_config(workspace, ready_config)
        comparison_hash, spec_hash = validation_hashes(ready_config, active_spec)
        sync_validation_activation(
            workspace.db_path,
            workspace.competition_id,
            ready_config,
            active_spec,
            workspace.validation_path(name),
            comparison_hash,
            spec_hash,
        )
    except Exception:
        save_validation_spec(workspace, spec)
        save_arena_config(workspace, config)
        raise

    return ready_config, active_spec
