from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone

import pytest
from typer.testing import CliRunner

from arenapilot.cli_entry import app
from arenapilot.db import get_experiment
from arenapilot.experiments import create_experiment, freeze_experiment
from arenapilot.intake import configure_intake
from arenapilot.memory import (
    MemoryError,
    approve_finding,
    create_finding,
    failure_modes,
    fingerprint_summary,
    knowledge_detail,
    learn,
    record_observation_evidence,
    retrieve_knowledge,
    set_competition_fingerprint,
)
from arenapilot.memory_schema import (
    MEMORY_SCHEMA_VERSION,
    initialize_workspace_memory_schema,
    read_workspace_memory_schema_version,
)
from arenapilot.models import MetricDirection, PredictionType, SplitType, TaskType
from arenapilot.validation import activate_validation, configure_validation
from arenapilot.workspace import create_workspace


runner = CliRunner()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _workspace_with_verified_run(tmp_path, slug: str):
    workspace = create_workspace(f"kaggle:{slug}", tmp_path / slug)
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
    spec = create_experiment(
        workspace,
        title="baseline",
        hypothesis="A baseline establishes the comparison floor.",
        model_family="catboost",
    )
    freeze_experiment(workspace, spec.id)
    experiment = get_experiment(workspace.db_path, workspace.competition_id, spec.id)
    assert experiment is not None
    with sqlite3.connect(workspace.db_path) as connection:
        connection.execute(
            """
            INSERT INTO runs(
                id, competition_id, experiment_id, name, status,
                backend, spec_hash, created_at, verified_at
            ) VALUES (?, ?, ?, 'run001', 'verified', 'local', ?, ?, ?)
            """,
            (
                f"run_{slug}",
                workspace.competition_id,
                experiment["id"],
                experiment["config_hash"],
                _now(),
                _now(),
            ),
        )
    initialize_workspace_memory_schema(workspace.db_path)
    return workspace


def _fingerprint(workspace):
    return set_competition_fingerprint(
        workspace,
        observed={
            "dataset": {
                "rows_bucket": "large",
                "columns_bucket": "medium",
                "high_cardinality": True,
            },
            "modalities": {"tabular": True},
        },
        inferred={
            "structure": {"temporal": False, "grouped": False},
            "shift": {"train_test_shift": "low"},
        },
        source="test",
    )


def _approved_frequency_finding(workspace, conclusion: str, outcome: str):
    evidence = record_observation_evidence(
        workspace,
        subject_type="technique",
        subject_key="frequency_encoding",
        outcome=outcome,
        summary=f"frequency encoding evidence: {outcome}",
        run_name="run001",
        strength=1,
    )
    finding = create_finding(
        workspace,
        subject_type="technique",
        subject_key="frequency_encoding",
        conclusion=conclusion,
        summary=f"frequency encoding is {conclusion} in this competition",
        evidence_names=[str(evidence["name"])],
        confidence="medium" if conclusion != "inconclusive" else "low",
    )
    return approve_finding(workspace, str(finding["name"]))


def test_workspace_memory_schema_is_versioned_inside_arena_db(tmp_path) -> None:
    workspace = _workspace_with_verified_run(tmp_path, "memory-schema")
    assert read_workspace_memory_schema_version(workspace.db_path) == MEMORY_SCHEMA_VERSION == 1
    with sqlite3.connect(workspace.db_path) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
    assert {
        "memory_schema_meta",
        "competition_fingerprints",
        "memory_evidence",
        "findings",
        "finding_evidence",
    } <= tables


def test_fingerprint_preserves_observed_vs_inferred_provenance(tmp_path) -> None:
    workspace = _workspace_with_verified_run(tmp_path, "fingerprint")
    first = _fingerprint(workspace)
    second = _fingerprint(workspace)
    assert first["fingerprint_hash"] == second["fingerprint_hash"]

    shown = fingerprint_summary(workspace)
    assert shown is not None
    payload = shown["fingerprint"]
    assert payload["task"]["type"] == "binary_classification"
    assert payload["observed"]["dataset"]["high_cardinality"] is True
    assert payload["inferred"]["structure"]["grouped"] is False


def test_finding_requires_matching_evidence_and_explicit_approval(tmp_path) -> None:
    workspace = _workspace_with_verified_run(tmp_path, "finding")
    _fingerprint(workspace)
    evidence = record_observation_evidence(
        workspace,
        subject_type="failure_mode",
        subject_key="group_leakage",
        outcome="warning",
        summary="Repeated customer IDs cross validation folds.",
        run_name="run001",
        strength=1,
    )

    with pytest.raises(MemoryError, match="finding requires evidence"):
        create_finding(
            workspace,
            subject_type="failure_mode",
            subject_key="group_leakage",
            conclusion="supported",
            summary="Group leakage is present.",
            evidence_names=[],
        )

    finding = create_finding(
        workspace,
        subject_type="failure_mode",
        subject_key="group_leakage",
        conclusion="supported",
        summary="Group leakage is present.",
        evidence_names=[str(evidence["name"])],
        confidence="medium",
    )
    assert finding["status"] == "candidate"
    approved = approve_finding(workspace, str(finding["name"]))
    assert approved["status"] == "approved"


def test_cross_competition_knowledge_versions_and_preserves_contradiction(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ARENAPILOT_HOME", str(tmp_path / "home"))

    first = _workspace_with_verified_run(tmp_path, "comp-a")
    _fingerprint(first)
    _approved_frequency_finding(first, "supported", "positive")
    v1 = learn(first)[0]
    assert v1["version"] == 1
    assert v1["confidence"] == "low"
    assert v1["independent_competitions"] == 1

    second = _workspace_with_verified_run(tmp_path, "comp-b")
    _fingerprint(second)
    _approved_frequency_finding(second, "supported", "positive")
    v2 = learn(second)[0]
    assert v2["version"] == 2
    assert v2["confidence"] == "medium"
    assert v2["positive_count"] == 2
    assert v2["independent_competitions"] == 2
    assert v2["supersedes_id"] == v1["id"]

    third = _workspace_with_verified_run(tmp_path, "comp-c")
    _fingerprint(third)
    _approved_frequency_finding(third, "rejected", "negative")
    v3 = learn(third)[0]
    assert v3["version"] == 3
    assert v3["confidence"] == "medium"
    assert v3["positive_count"] == 2
    assert v3["negative_count"] == 1
    assert v3["independent_competitions"] == 3
    assert v3["directional_consistency"] == pytest.approx(2 / 3)

    retrieved = retrieve_knowledge(third, query="frequency", limit=5)
    assert len(retrieved) == 1
    assert retrieved[0]["key"] == "frequency_encoding"
    assert retrieved[0]["version"] == 3
    assert retrieved[0]["contradictory_evidence"] == 1
    assert retrieved[0]["fingerprint_similarity"] == pytest.approx(1.0)

    detail = knowledge_detail("technique", "frequency_encoding")
    assert len(detail["evidence"]) == 3
    assert {item["source_competition_slug"] for item in detail["evidence"]} == {
        "comp-a",
        "comp-b",
        "comp-c",
    }


def test_failure_mode_registry_contains_validation_failures() -> None:
    keys = {item["key"] for item in failure_modes()}
    assert {"temporal_leakage", "group_leakage", "target_leakage", "invalid_cv"} <= keys


def test_memory_cli_fingerprint_and_failure_registry_json(tmp_path, monkeypatch) -> None:
    workspace = _workspace_with_verified_run(tmp_path, "cli-memory")
    monkeypatch.chdir(workspace.root)

    result = runner.invoke(
        app,
        [
            "fingerprint",
            "set",
            "--observed-json",
            '{"dataset":{"high_cardinality":true}}',
            "--inferred-json",
            '{"structure":{"grouped":false}}',
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["ok"] is True
    assert payload["fingerprint"]["observed"]["dataset"]["high_cardinality"] is True

    result = runner.invoke(app, ["failure", "list", "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert any(item["key"] == "group_leakage" for item in payload["failure_modes"])
