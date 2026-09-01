import pytest
from pydantic import ValidationError

from arenapilot.models import (
    ArenaConfig,
    ExperimentSpec,
    SplitConfig,
    SplitType,
    SubmissionConfig,
    ValidationSpec,
)


def test_arena_config_accepts_ready_contract() -> None:
    config = ArenaConfig.model_validate(
        {
            "schema_version": 1,
            "competition": {"platform": "kaggle", "slug": "demo", "status": "ready"},
            "task": {"type": "binary_classification", "target": "target"},
            "metric": {"name": "roc_auc", "direction": "maximize"},
            "validation": {"active": "val-v1"},
            "tracking": {"backend": "mlflow", "experiment_name": "demo"},
        }
    )
    assert config.competition.slug == "demo"
    assert config.workspace.default_seed == 42


def test_arena_config_allows_explicit_draft_before_intake() -> None:
    config = ArenaConfig.model_validate(
        {
            "competition": {"platform": "kaggle", "slug": "demo", "status": "draft"},
            "tracking": {"backend": "mlflow", "experiment_name": "demo"},
        }
    )
    assert config.task is None
    assert config.metric is None
    assert config.validation.active is None


def test_ready_arena_config_requires_task_metric_and_validation() -> None:
    with pytest.raises(ValidationError):
        ArenaConfig.model_validate(
            {
                "competition": {"platform": "kaggle", "slug": "demo", "status": "ready"},
                "tracking": {"backend": "mlflow", "experiment_name": "demo"},
            }
        )


def test_group_split_requires_group_column() -> None:
    with pytest.raises(ValidationError):
        SplitConfig(type=SplitType.GROUP_KFOLD)


def test_submission_budget_rejects_impossible_daily_budget() -> None:
    with pytest.raises(ValidationError):
        SubmissionConfig(daily_budget=5, total_budget=3)


def test_experiment_requires_stable_ids() -> None:
    with pytest.raises(ValidationError):
        ExperimentSpec.model_validate(
            {
                "id": "experiment-1",
                "title": "baseline",
                "hypothesis": "A baseline should establish the comparison floor.",
                "validation": "val-v1",
                "model": {"family": "catboost"},
            }
        )


def test_validation_draft_can_be_incomplete() -> None:
    spec = ValidationSpec.model_validate(
        {
            "id": "val-v1",
            "status": "draft",
            "reason": "initial_validation",
            "split": {"type": "kfold"},
        }
    )
    assert spec.metric is None
    assert spec.prediction is None


def test_active_validation_requires_metric_and_prediction() -> None:
    with pytest.raises(ValidationError):
        ValidationSpec.model_validate(
            {
                "id": "val-v1",
                "status": "active",
                "reason": "initial_validation",
                "split": {"type": "kfold"},
            }
        )
