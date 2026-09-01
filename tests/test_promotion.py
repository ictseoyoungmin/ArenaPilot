from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from arenapilot.cli_entry import app
from arenapilot.intake import configure_intake
from arenapilot.memory import MemoryError, set_competition_fingerprint
from arenapilot.memory_store import (
    create_knowledge_version,
    initialize_knowledge_database,
    knowledge_db_path,
)
from arenapilot.models import MetricDirection, PredictionType, SplitType, TaskType
from arenapilot.promotion import (
    approve_knowledge,
    assess_knowledge,
    deprecate_knowledge,
    deprecate_technique,
    knowledge_history,
    ranked_knowledge,
    register_technique,
)
from arenapilot.promotion_store import (
    initialize_promotion_database,
    upsert_global_independence_profile,
)
from arenapilot.validation import activate_validation, configure_validation
from arenapilot.workspace import create_workspace


runner = CliRunner()


def _ready_workspace(tmp_path, slug: str):
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
    set_competition_fingerprint(
        workspace,
        observed={
            "dataset": {"high_cardinality": True, "rows_bucket": "large"},
            "modalities": {"tabular": True},
        },
        inferred={"structure": {"grouped": False, "temporal": False}},
        source="test",
    )
    return workspace


def _fingerprint(*, high_cardinality: bool = True, grouped: bool = False):
    return {
        "schema_version": 1,
        "task": {"type": "binary_classification"},
        "metric": {"name": "roc_auc", "direction": "maximize"},
        "observed": {
            "dataset": {"high_cardinality": high_cardinality, "rows_bucket": "large"},
            "modalities": {"tabular": True},
        },
        "inferred": {"structure": {"grouped": grouped, "temporal": False}},
    }


def _evidence(competition_id: str, slug: str, *, conclusion: str = "supported", strength: int = 2, fingerprint=None):
    return {
        "source_competition_id": competition_id,
        "source_competition_slug": slug,
        "source_finding_id": f"finding_{competition_id}",
        "source_finding_name": "finding001",
        "conclusion": conclusion,
        "strength": strength,
        "fingerprint": fingerprint or _fingerprint(),
    }


def _create_version(path, evidence, *, key: str = "frequency_encoding"):
    positive = sum(1 for item in evidence if item["conclusion"] == "supported")
    negative = sum(1 for item in evidence if item["conclusion"] == "rejected")
    directional = positive + negative
    consistency = max(positive, negative) / directional if directional else None
    return create_knowledge_version(
        path,
        kind="technique",
        key=key,
        title=key.replace("_", " ").title(),
        summary=f"{positive} supported / {negative} rejected",
        applicability={"task_types": ["binary_classification"], "metric_names": ["roc_auc"]},
        confidence="medium" if len(evidence) >= 2 else "low",
        independent_competitions=len({item["source_competition_id"] for item in evidence}),
        positive_count=positive,
        neutral_count=0,
        negative_count=negative,
        directional_consistency=consistency,
        evidence=evidence,
        supersedes_id=None,
    )


def _profile(path, competition_id: str, slug: str, group: str, *, relation: str = "independent"):
    upsert_global_independence_profile(
        path,
        competition_id=competition_id,
        competition_slug=slug,
        independence_key=group,
        dataset_key=group if relation == "same_dataset" else None,
        relation=relation,
        parent_competition_slug="comp-a" if relation == "derived" else None,
        source="test",
    )


def test_independence_groups_block_false_high_confidence_and_control_supersession(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ARENAPILOT_HOME", str(tmp_path / "home"))
    path = initialize_promotion_database(knowledge_db_path())

    evidence = [
        _evidence("cmp-a", "comp-a"),
        _evidence("cmp-b", "comp-b"),
        _evidence("cmp-c", "comp-c"),
        _evidence("cmp-d", "comp-d"),
    ]
    v1 = _create_version(path, evidence)
    _profile(path, "cmp-a", "comp-a", "dataset-family-a")
    _profile(path, "cmp-b", "comp-b", "dataset-family-a", relation="derived")
    _profile(path, "cmp-c", "comp-c", "dataset-family-c")
    _profile(path, "cmp-d", "comp-d", "dataset-family-d")

    assessment = assess_knowledge("technique", "frequency_encoding")
    assert assessment["raw_competitions"] == 4
    assert assessment["independent_units"] == 3
    assert assessment["directional_units"] == 3
    assert assessment["directional_consistency"] == pytest.approx(1.0)

    register_technique(
        "frequency_encoding",
        category="categorical_encoding",
        description="Encode category frequency without target statistics.",
    )
    with pytest.raises(MemoryError, match="high requires >=4 independent directional units"):
        approve_knowledge("technique", "frequency_encoding", confidence="high")

    evidence.append(_evidence("cmp-e", "comp-e"))
    v2 = create_knowledge_version(
        path,
        kind="technique",
        key="frequency_encoding",
        title="Frequency Encoding",
        summary="five raw competitions with four independent evidence units",
        applicability={"task_types": ["binary_classification"], "metric_names": ["roc_auc"]},
        confidence="medium",
        independent_competitions=5,
        positive_count=5,
        neutral_count=0,
        negative_count=0,
        directional_consistency=1.0,
        evidence=evidence,
        supersedes_id=str(v1["id"]),
    )
    _profile(path, "cmp-e", "comp-e", "dataset-family-e")

    approved = approve_knowledge(
        "technique",
        "frequency_encoding",
        confidence="high",
        reason="Four independent directional units with consistent results.",
    )
    assert approved["knowledge"]["version"] == v2["version"] == 2
    assert approved["promotion"]["status"] == "approved"
    assert approved["promotion"]["approved_confidence"] == "high"
    assert approved["assessment"]["independent_units"] == 4

    v3 = create_knowledge_version(
        path,
        kind="technique",
        key="frequency_encoding",
        title="Frequency Encoding",
        summary="new version preserving the same evidence while refining applicability",
        applicability={"task_types": ["binary_classification"], "metric_names": ["roc_auc"]},
        confidence="medium",
        independent_competitions=5,
        positive_count=5,
        neutral_count=0,
        negative_count=0,
        directional_consistency=1.0,
        evidence=evidence,
        supersedes_id=str(v2["id"]),
    )
    promoted_v3 = approve_knowledge(
        "technique", "frequency_encoding", confidence="medium", reason="Refined applicability."
    )
    assert promoted_v3["knowledge"]["version"] == v3["version"] == 3

    history = knowledge_history("technique", "frequency_encoding")
    assert history[1]["promotion_status"] == "superseded"
    assert history[1]["superseded_by_id"] == v3["id"]
    assert history[2]["promotion_status"] == "approved"

    deprecated = deprecate_knowledge(
        "technique", "frequency_encoding", "Technique semantics changed; issue a new registry key."
    )
    assert deprecated["promotion"]["status"] == "deprecated"


def test_high_confidence_rejects_major_contradiction(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ARENAPILOT_HOME", str(tmp_path / "home"))
    path = initialize_promotion_database(knowledge_db_path())
    evidence = [
        _evidence("a", "a", strength=2),
        _evidence("b", "b", strength=2),
        _evidence("c", "c", strength=2),
        _evidence("d", "d", strength=3),
        _evidence("e", "e", conclusion="rejected", strength=2),
    ]
    _create_version(path, evidence, key="target_encoding")
    for item in evidence:
        _profile(path, item["source_competition_id"], item["source_competition_slug"], f"group-{item['source_competition_id']}")
    register_technique("target_encoding", category="categorical_encoding")

    assessment = assess_knowledge("technique", "target_encoding")
    assert assessment["independent_units"] == 5
    assert assessment["directional_consistency"] == pytest.approx(0.8)
    assert assessment["major_contradiction"] is True
    with pytest.raises(MemoryError, match="major unresolved contradiction"):
        approve_knowledge("technique", "target_encoding", confidence="high")


def test_ranked_retrieval_prefers_approved_observed_match_and_hides_deprecated_technique(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ARENAPILOT_HOME", str(tmp_path / "home"))
    workspace = _ready_workspace(tmp_path, "current")
    path = initialize_promotion_database(knowledge_db_path())

    matching = [_evidence("match-a", "match-a"), _evidence("match-b", "match-b")]
    _create_version(path, matching, key="frequency_encoding")
    _profile(path, "match-a", "match-a", "match-a")
    _profile(path, "match-b", "match-b", "match-b")
    register_technique("frequency_encoding", category="categorical_encoding")
    approve_knowledge("technique", "frequency_encoding", confidence="medium")

    mismatched = [
        _evidence(
            "other-a",
            "other-a",
            fingerprint=_fingerprint(high_cardinality=False, grouped=True),
        )
    ]
    _create_version(path, mismatched, key="ordinal_encoding")
    _profile(path, "other-a", "other-a", "other-a")
    register_technique("ordinal_encoding", category="categorical_encoding")

    ranked = ranked_knowledge(workspace)
    assert ranked[0]["key"] == "frequency_encoding"
    assert ranked[0]["status"] == "approved"
    assert ranked[0]["observed_similarity"] == pytest.approx(1.0)
    assert ranked[0]["independent_units"] == 2

    deprecate_technique("frequency_encoding", "Superseded by a safer canonical implementation.")
    ranked_after = ranked_knowledge(workspace)
    assert all(item["key"] != "frequency_encoding" for item in ranked_after)
    assert ranked_after[0]["key"] == "ordinal_encoding"


def test_promotion_cli_exposes_independence_and_technique_registry(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ARENAPILOT_HOME", str(tmp_path / "home"))
    workspace = _ready_workspace(tmp_path, "cli-promotion")
    monkeypatch.chdir(workspace.root)

    result = runner.invoke(
        app,
        [
            "independence",
            "set",
            "--group",
            "playground-synthetic-family",
            "--dataset-key",
            "synthetic-v1",
            "--relation",
            "same_dataset",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["ok"] is True
    assert payload["independence"]["independence_key"] == "playground-synthetic-family"
    assert payload["independence"]["relation"] == "same_dataset"

    result = runner.invoke(
        app,
        [
            "technique",
            "register",
            "frequency_encoding",
            "--category",
            "categorical_encoding",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["technique"]["key"] == "frequency_encoding"
    assert payload["technique"]["status"] == "active"
