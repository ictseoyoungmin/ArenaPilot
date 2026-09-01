import pytest
from pydantic import ValidationError

from arenapilot.models import (
    ArenaConfig,
    ExperimentSpec,
    SplitConfig,
    SplitType,
    SubmissionConfig,
)


def test_arena_config_accepts_v0_contract() -> None:
    config = ArenaConfig.model_validate(
        {
            "schema_version": 1,
            "competition": {"platform": "kaggle", "slug": "demo"},
            "task": {"type": "binary_classification", "target": "target"},
            "metric": {"name": "roc_auc", "direction": "maximize"},
            "validation": {"active": "val-v1"},
            "tracking": {"backend": "mlflow", "experiment_name": "demo"},
        }
    )
    assert config.competition.slug == "demo"
    assert config.workspace.default_seed == 42


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
