from __future__ import annotations

import statistics
from typing import Any

from .db import get_experiment
from .models import ExperimentSpec, MetricDirection
from .runstore import get_canonical_run_for_experiment
from .tracking import mlflow_run_summary
from .workspace import Workspace, WorkspaceError, load_arena_config, load_experiment_spec


class ComparisonError(WorkspaceError):
    pass


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    if isinstance(value, dict):
        for key in sorted(value):
            name = f"{prefix}.{key}" if prefix else str(key)
            item = value[key]
            if isinstance(item, dict):
                flattened.update(_flatten(item, name))
            else:
                flattened[name] = item
    elif prefix:
        flattened[prefix] = value
    return flattened


def _comparable_config(spec: ExperimentSpec) -> dict[str, Any]:
    return {
        "model": spec.model,
        "pipeline": spec.pipeline,
        "seed": spec.seed.model_dump(mode="json"),
        "runtime": spec.runtime.model_dump(mode="json"),
    }


def _config_diff(baseline: ExperimentSpec, candidate: ExperimentSpec) -> list[dict[str, Any]]:
    left = _flatten(_comparable_config(baseline))
    right = _flatten(_comparable_config(candidate))
    changes: list[dict[str, Any]] = []
    for key in sorted(set(left) | set(right)):
        if left.get(key) != right.get(key):
            changes.append(
                {
                    "path": key,
                    "baseline": left.get(key),
                    "candidate": right.get(key),
                }
            )
    return changes


def _fold_values(metrics: dict[str, float], metric_name: str) -> dict[int, float]:
    suffix = f"_{metric_name}"
    result: dict[int, float] = {}
    for key, value in metrics.items():
        if not key.startswith("fold_") or not key.endswith(suffix):
            continue
        middle = key[len("fold_") : -len(suffix)]
        if middle.isdigit():
            result[int(middle)] = float(value)
    return result


def _stability(values: dict[int, float]) -> float | None:
    ordered = [values[index] for index in sorted(values)]
    if not ordered:
        return None
    if len(ordered) == 1:
        return 0.0
    return statistics.pstdev(ordered)


def _require_canonical_tracking(
    workspace: Workspace,
    experiment_name: str,
    metric_name: str,
) -> tuple[dict[str, object], dict[str, object]]:
    canonical = get_canonical_run_for_experiment(
        workspace.db_path,
        workspace.competition_id,
        experiment_name,
    )
    if canonical is None or canonical.get("status") != "verified":
        raise ComparisonError(f"EXPERIMENT_NOT_COMPARABLE: {experiment_name} has no VERIFIED canonical run")
    mlflow_run_id = canonical.get("mlflow_run_id")
    if not mlflow_run_id:
        raise ComparisonError(f"EXPERIMENT_NOT_COMPARABLE: {experiment_name} canonical run is not tracked")
    tracking = mlflow_run_summary(str(mlflow_run_id), metric_name)
    if tracking.get("primary_metric") is None:
        raise ComparisonError(
            f"EXPERIMENT_NOT_COMPARABLE: {experiment_name} has no tracked primary metric {metric_name}"
        )
    return canonical, tracking


def compare_experiments(
    workspace: Workspace,
    baseline_name: str,
    candidate_name: str,
) -> dict[str, object]:
    if baseline_name == candidate_name:
        raise ComparisonError("experiments must be distinct")

    baseline_row = get_experiment(workspace.db_path, workspace.competition_id, baseline_name)
    candidate_row = get_experiment(workspace.db_path, workspace.competition_id, candidate_name)
    if baseline_row is None:
        raise ComparisonError(f"experiment not found: {baseline_name}")
    if candidate_row is None:
        raise ComparisonError(f"experiment not found: {candidate_name}")

    baseline_domain = str(baseline_row["comparison_domain_hash"])
    candidate_domain = str(candidate_row["comparison_domain_hash"])
    if baseline_domain != candidate_domain:
        raise ComparisonError(
            "COMPARISON_DOMAIN_MISMATCH: "
            f"{baseline_name} ({baseline_row['validation_id']}) and "
            f"{candidate_name} ({candidate_row['validation_id']}) use incompatible validation domains"
        )

    config = load_arena_config(workspace)
    if config.metric is None:
        raise ComparisonError("competition metric is not configured")
    metric_name = config.metric.name
    direction = config.metric.direction

    baseline_run, baseline_tracking = _require_canonical_tracking(
        workspace,
        baseline_name,
        metric_name,
    )
    candidate_run, candidate_tracking = _require_canonical_tracking(
        workspace,
        candidate_name,
        metric_name,
    )

    baseline_value = float(baseline_tracking["primary_metric"])
    candidate_value = float(candidate_tracking["primary_metric"])
    raw_delta = candidate_value - baseline_value
    direction_multiplier = 1.0 if direction == MetricDirection.MAXIMIZE else -1.0
    normalized_delta = raw_delta * direction_multiplier

    baseline_metrics = {
        str(key): float(value)
        for key, value in dict(baseline_tracking.get("metrics") or {}).items()
    }
    candidate_metrics = {
        str(key): float(value)
        for key, value in dict(candidate_tracking.get("metrics") or {}).items()
    }
    baseline_folds = _fold_values(baseline_metrics, metric_name)
    candidate_folds = _fold_values(candidate_metrics, metric_name)
    common_folds = sorted(set(baseline_folds) & set(candidate_folds))
    fold_deltas = [
        {
            "fold": fold,
            "baseline": baseline_folds[fold],
            "candidate": candidate_folds[fold],
            "raw_delta": candidate_folds[fold] - baseline_folds[fold],
            "direction_normalized_delta": (
                candidate_folds[fold] - baseline_folds[fold]
            )
            * direction_multiplier,
        }
        for fold in common_folds
    ]

    baseline_std = _stability(baseline_folds)
    candidate_std = _stability(candidate_folds)
    stability_delta = (
        candidate_std - baseline_std
        if baseline_std is not None and candidate_std is not None
        else None
    )

    baseline_duration = baseline_metrics.get("duration_seconds")
    candidate_duration = candidate_metrics.get("duration_seconds")
    duration_delta = (
        candidate_duration - baseline_duration
        if baseline_duration is not None and candidate_duration is not None
        else None
    )

    baseline_spec = load_experiment_spec(workspace, baseline_name)
    candidate_spec = load_experiment_spec(workspace, candidate_name)

    if normalized_delta > 0:
        outcome = "improved"
    elif normalized_delta < 0:
        outcome = "regressed"
    else:
        outcome = "tied"

    return {
        "baseline": {
            "experiment": baseline_name,
            "run": baseline_run["name"],
            "mlflow_run_id": baseline_run["mlflow_run_id"],
            "primary_metric": baseline_value,
        },
        "candidate": {
            "experiment": candidate_name,
            "run": candidate_run["name"],
            "mlflow_run_id": candidate_run["mlflow_run_id"],
            "primary_metric": candidate_value,
        },
        "comparison_domain": baseline_domain,
        "validation": str(baseline_row["validation_id"]),
        "metric": {
            "name": metric_name,
            "direction": direction.value,
            "raw_delta": raw_delta,
            "direction_normalized_delta": normalized_delta,
            "outcome": outcome,
        },
        "folds": {
            "common_count": len(common_folds),
            "deltas": fold_deltas,
            "baseline_std": baseline_std,
            "candidate_std": candidate_std,
            "std_delta": stability_delta,
        },
        "runtime": {
            "baseline_seconds": baseline_duration,
            "candidate_seconds": candidate_duration,
            "delta_seconds": duration_delta,
        },
        "config_changes": _config_diff(baseline_spec, candidate_spec),
    }
