from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from arenapilot.comparison import ComparisonError, compare_experiments
from arenapilot.experiments import create_experiment, freeze_experiment
from arenapilot.intake import configure_intake
from arenapilot.models import MetricDirection, PredictionType, SplitType, TaskType
from arenapilot.runs import run_local_experiment
from arenapilot.validation import activate_validation, configure_validation
from arenapilot.workspace import (
    create_workspace,
    load_experiment_spec,
    save_experiment_spec,
)


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


def _trainer(metric: float, folds: list[float], duration: float) -> str:
    return f'''\
import json
import os
from pathlib import Path

out = Path(os.environ["ARENA_OUTPUT_DIR"])
(out / "predictions.parquet").write_bytes(b"predictions")
(out / "oof.parquet").write_bytes(b"oof")
(out / "result.json").write_text(json.dumps({{
    "status": "success",
    "primary_metric": {{"name": "roc_auc", "value": {metric}}},
    "duration_seconds": {duration}
}}), encoding="utf-8")
(out / "metrics.json").write_text(json.dumps({{"roc_auc": {metric}}}), encoding="utf-8")
(out / "fold_metrics.json").write_text(json.dumps({{
    "folds": [
        {{"fold": index, "roc_auc": value}}
        for index, value in enumerate({folds!r})
    ]
}}), encoding="utf-8")
'''


def _run_pair(tmp_path):
    workspace = _ready_workspace(tmp_path)
    baseline = create_experiment(
        workspace,
        title="baseline",
        hypothesis="A baseline establishes the comparison floor.",
        model_family="catboost",
    )
    baseline_spec = load_experiment_spec(workspace, baseline.id)
    baseline_spec.model["params"] = {"depth": 6}
    save_experiment_spec(workspace, baseline_spec)
    freeze_experiment(workspace, baseline.id)

    (workspace.root / "src" / "train.py").write_text(
        _trainer(0.8100, [0.80, 0.81, 0.82, 0.81, 0.81], 10.0),
        encoding="utf-8",
    )
    run_local_experiment(workspace, baseline.id)

    candidate = create_experiment(
        workspace,
        title="deeper-catboost",
        hypothesis="A deeper CatBoost improves validation AUC.",
        model_family="catboost",
        parent=baseline.id,
    )
    candidate_spec = load_experiment_spec(workspace, candidate.id)
    candidate_spec.model["params"] = {"depth": 8}
    candidate_spec.pipeline["frequency_encoding"] = True
    save_experiment_spec(workspace, candidate_spec)
    freeze_experiment(workspace, candidate.id)

    (workspace.root / "src" / "train.py").write_text(
        _trainer(0.8230, [0.82, 0.823, 0.824, 0.825, 0.823], 12.5),
        encoding="utf-8",
    )
    run_local_experiment(workspace, candidate.id)
    return workspace, baseline.id, candidate.id


def test_compare_uses_canonical_mlflow_metrics_and_config_diff(tmp_path) -> None:
    workspace, baseline, candidate = _run_pair(tmp_path)

    result = compare_experiments(workspace, baseline, candidate)

    assert result["baseline"]["run"] == "run001"
    assert result["candidate"]["run"] == "run002"
    assert result["metric"]["name"] == "roc_auc"
    assert result["metric"]["direction"] == "maximize"
    assert result["metric"]["raw_delta"] == pytest.approx(0.013)
    assert result["metric"]["direction_normalized_delta"] == pytest.approx(0.013)
    assert result["metric"]["outcome"] == "improved"
    assert result["folds"]["common_count"] == 5
    assert result["folds"]["candidate_std"] < result["folds"]["baseline_std"]
    assert result["runtime"]["delta_seconds"] == pytest.approx(2.5)

    changes = {item["path"]: item for item in result["config_changes"]}
    assert changes["model.params.depth"] == {
        "path": "model.params.depth",
        "baseline": 6,
        "candidate": 8,
    }
    assert changes["pipeline.frequency_encoding"]["baseline"] is None
    assert changes["pipeline.frequency_encoding"]["candidate"] is True


def test_compare_rejects_different_comparison_domains_before_metric_delta(tmp_path) -> None:
    workspace, baseline, candidate = _run_pair(tmp_path)
    with sqlite3.connect(workspace.db_path) as connection:
        connection.execute(
            "UPDATE experiments SET comparison_domain_hash = ? WHERE competition_id = ? AND name = ?",
            ("different-domain", workspace.competition_id, candidate),
        )

    with pytest.raises(ComparisonError, match="COMPARISON_DOMAIN_MISMATCH"):
        compare_experiments(workspace, baseline, candidate)


def test_compare_requires_verified_canonical_runs(tmp_path) -> None:
    workspace = _ready_workspace(tmp_path)
    baseline = create_experiment(
        workspace,
        title="baseline",
        hypothesis="A baseline establishes the comparison floor.",
        model_family="catboost",
    )
    freeze_experiment(workspace, baseline.id)
    candidate = create_experiment(
        workspace,
        title="candidate",
        hypothesis="A candidate should improve the baseline.",
        model_family="catboost",
        parent=baseline.id,
    )
    freeze_experiment(workspace, candidate.id)

    with pytest.raises(ComparisonError, match="EXPERIMENT_NOT_COMPARABLE"):
        compare_experiments(workspace, baseline.id, candidate.id)
