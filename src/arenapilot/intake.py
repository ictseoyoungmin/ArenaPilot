from __future__ import annotations

from .db import sync_competition
from .models import ArenaConfig, MetricConfig, MetricDirection, TaskConfig, TaskType
from .workspace import Workspace, WorkspaceError, load_arena_config, save_arena_config


def configure_intake(
    workspace: Workspace,
    *,
    task_type: TaskType,
    target: str,
    metric_name: str,
    metric_direction: MetricDirection,
) -> ArenaConfig:
    config = load_arena_config(workspace)
    if config.competition.status != "draft":
        raise WorkspaceError("competition intake is immutable after validation activation")

    payload = config.model_dump(mode="json")
    payload["task"] = TaskConfig(type=task_type, target=target).model_dump(mode="json")
    payload["metric"] = MetricConfig(
        name=metric_name,
        direction=metric_direction,
    ).model_dump(mode="json")
    updated = ArenaConfig.model_validate(payload)

    save_arena_config(workspace, updated)
    try:
        sync_competition(workspace.db_path, workspace.competition_id, updated)
    except Exception:
        save_arena_config(workspace, config)
        raise

    return updated
