import yaml
import pytest

from arenapilot.db import get_experiment
from arenapilot.experiments import (
    ExperimentError,
    create_experiment,
    experiment_lineage,
    freeze_experiment,
    list_experiment_summaries,
    show_experiment,
)
from arenapilot.intake import configure_intake
from arenapilot.models import MetricDirection, PredictionType, SplitType, TaskType
from arenapilot.validation import activate_validation, configure_validation
from arenapilot.workspace import create_workspace


def _ready_workspace(tmp_path):
    workspace = create_workspace("kaggle:demo", tmp_path / "demo")
    configure_intake(
        workspace,
        task_type=TaskType.BINARY_CLASSIFICATION,
        target="target",
        metric_name="roc_auc",
        metric_direction=MetricDirection.MAXIMIZE,
    )
    configure_validation(
        workspace,
        "val-v1",
        split_type=SplitType.STRATIFIED_KFOLD,
        prediction_type=PredictionType.PROBABILITY,
    )
    activate_validation(workspace, "val-v1")
    return workspace


def test_experiment_creation_requires_ready_competition(tmp_path) -> None:
    workspace = create_workspace("kaggle:demo", tmp_path / "demo")
    with pytest.raises(ExperimentError):
        create_experiment(
            workspace,
            title="baseline",
            hypothesis="A baseline establishes the comparison floor.",
            model_family="catboost",
        )


def test_experiment_is_bound_to_active_validation_and_auto_numbered(tmp_path) -> None:
    workspace = _ready_workspace(tmp_path)
    first = create_experiment(
        workspace,
        title="baseline",
        hypothesis="A baseline establishes the comparison floor.",
        model_family="catboost",
    )
    second = create_experiment(
        workspace,
        title="lightgbm-baseline",
        hypothesis="LightGBM provides an alternate baseline.",
        model_family="lightgbm",
    )

    assert first.id == "exp001"
    assert second.id == "exp002"
    assert first.validation == "val-v1"
    row = get_experiment(workspace.db_path, workspace.competition_id, "exp001")
    assert row is not None
    assert row["status"] == "draft"
    assert row["validation_id"] == "val-v1"


def test_freeze_persists_snapshot_and_detects_spec_mutation(tmp_path) -> None:
    workspace = _ready_workspace(tmp_path)
    spec = create_experiment(
        workspace,
        title="baseline",
        hypothesis="A baseline establishes the comparison floor.",
        model_family="catboost",
    )

    record, snapshot = freeze_experiment(workspace, spec.id)
    assert record["status"] == "frozen"
    assert record["config_hash"]
    assert snapshot.is_file()

    payload = yaml.safe_load(workspace.experiment_path(spec.id).read_text(encoding="utf-8"))
    payload["model"]["params"]["depth"] = 10
    workspace.experiment_path(spec.id).write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )

    shown = show_experiment(workspace, spec.id)
    assert shown["spec_integrity"] is False
    with pytest.raises(ExperimentError, match="FROZEN_SPEC_MODIFIED"):
        freeze_experiment(workspace, spec.id)


def test_parent_must_be_frozen_and_lineage_is_persisted(tmp_path) -> None:
    workspace = _ready_workspace(tmp_path)
    parent = create_experiment(
        workspace,
        title="baseline",
        hypothesis="A baseline establishes the comparison floor.",
        model_family="catboost",
    )

    with pytest.raises(ExperimentError, match="must be frozen"):
        create_experiment(
            workspace,
            title="frequency-encoding",
            hypothesis="Frequency encoding improves high-cardinality categories.",
            model_family="catboost",
            parent=parent.id,
        )

    freeze_experiment(workspace, parent.id)
    child = create_experiment(
        workspace,
        title="frequency-encoding",
        hypothesis="Frequency encoding improves high-cardinality categories.",
        model_family="catboost",
        parent=parent.id,
    )
    edges = experiment_lineage(workspace, child.id)

    assert edges == [
        {
            "parent": "exp001",
            "child": "exp002",
            "relation": "derived_from",
        }
    ]
    summaries = list_experiment_summaries(workspace)
    assert [item["id"] for item in summaries] == ["exp001", "exp002"]
