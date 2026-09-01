from __future__ import annotations

import json
from typing import Any

from .comparison import ComparisonError, compare_experiments
from .db import get_experiment
from .memory_store import (
    approve_finding_record,
    approved_findings,
    create_evidence_record,
    create_finding_record,
    create_fingerprint_record,
    create_knowledge_version,
    get_evidence,
    get_finding,
    initialize_knowledge_database,
    knowledge_db_path,
    knowledge_evidence,
    latest_fingerprint,
    latest_knowledge_item,
    list_evidence,
    list_findings,
    list_latest_knowledge,
)
from .runstore import get_run
from .submission_store import get_submission
from .workspace import Workspace, WorkspaceError, load_arena_config


class MemoryError(WorkspaceError):
    pass


FAILURE_MODES: dict[str, str] = {
    "temporal_leakage": "Validation allows information from the future into training or features.",
    "group_leakage": "Related entities cross train/validation boundaries and inflate validation.",
    "target_leakage": "Features directly or indirectly encode the target unavailable at inference.",
    "public_lb_overfit": "Decisions are overfit to public leaderboard feedback rather than robust CV evidence.",
    "train_test_shift": "Material train/test distribution shift weakens validation-to-test transfer.",
    "invalid_cv": "Cross-validation construction or OOF accounting does not match the evaluation contract.",
}

_ALLOWED_SUBJECT_TYPES = {"technique", "failure_mode", "pattern"}
_ALLOWED_OUTCOMES = {"positive", "neutral", "negative", "warning"}
_ALLOWED_CONCLUSIONS = {"supported", "rejected", "mixed", "inconclusive"}


def _decode_json(value: object) -> object:
    if isinstance(value, str):
        return json.loads(value)
    return value


def _validate_subject(subject_type: str, subject_key: str) -> None:
    if subject_type not in _ALLOWED_SUBJECT_TYPES:
        raise MemoryError(f"unsupported memory subject type: {subject_type}")
    if not subject_key.strip():
        raise MemoryError("memory subject key cannot be empty")
    if subject_type == "failure_mode" and subject_key not in FAILURE_MODES:
        raise MemoryError(f"unknown failure mode: {subject_key}")


def set_competition_fingerprint(
    workspace: Workspace,
    *,
    observed: dict[str, object] | None = None,
    inferred: dict[str, object] | None = None,
    source: str = "manual",
) -> dict[str, object]:
    config = load_arena_config(workspace)
    if config.task is None or config.metric is None:
        raise MemoryError("competition intake must be configured before fingerprinting")
    fingerprint: dict[str, object] = {
        "schema_version": 1,
        "task": {"type": config.task.type.value},
        "metric": {
            "name": config.metric.name,
            "direction": config.metric.direction.value,
        },
        "observed": observed or {},
        "inferred": inferred or {},
    }
    return create_fingerprint_record(
        workspace.db_path,
        competition_id=workspace.competition_id,
        fingerprint=fingerprint,
        source=source,
    )


def fingerprint_summary(workspace: Workspace) -> dict[str, object] | None:
    row = latest_fingerprint(workspace.db_path, workspace.competition_id)
    if row is None:
        return None
    return {
        **row,
        "fingerprint": _decode_json(row["fingerprint_json"]),
    }


def record_comparison_evidence(
    workspace: Workspace,
    *,
    subject_type: str,
    subject_key: str,
    baseline: str,
    candidate: str,
    summary: str,
) -> dict[str, object]:
    _validate_subject(subject_type, subject_key)
    try:
        comparison = compare_experiments(workspace, baseline, candidate)
    except ComparisonError as exc:
        raise MemoryError(str(exc)) from exc

    normalized = float(comparison["metric"]["direction_normalized_delta"])
    outcome = "positive" if normalized > 0 else "negative" if normalized < 0 else "neutral"
    baseline_run = get_run(
        workspace.db_path,
        workspace.competition_id,
        str(comparison["baseline"]["run"]),
    )
    candidate_run = get_run(
        workspace.db_path,
        workspace.competition_id,
        str(comparison["candidate"]["run"]),
    )
    baseline_exp = get_experiment(workspace.db_path, workspace.competition_id, baseline)
    candidate_exp = get_experiment(workspace.db_path, workspace.competition_id, candidate)
    assert baseline_run is not None and candidate_run is not None
    assert baseline_exp is not None and candidate_exp is not None

    fold_count = int(comparison["folds"]["common_count"])
    strength = 1 if fold_count > 0 else 0
    context = {
        "source": "experiment_comparison",
        "baseline": comparison["baseline"],
        "candidate": comparison["candidate"],
        "metric": comparison["metric"],
        "folds": {
            "common_count": fold_count,
            "baseline_std": comparison["folds"]["baseline_std"],
            "candidate_std": comparison["folds"]["candidate_std"],
        },
        "config_changes": comparison["config_changes"],
    }
    return create_evidence_record(
        workspace.db_path,
        competition_id=workspace.competition_id,
        subject_type=subject_type,
        subject_key=subject_key,
        outcome=outcome,
        effect=normalized,
        strength=strength,
        summary=summary,
        context=context,
        validation_domain_hash=str(comparison["comparison_domain"]),
        source_experiment_id=str(candidate_exp["id"]),
        source_run_id=str(candidate_run["id"]),
        reference_experiment_id=str(baseline_exp["id"]),
        reference_run_id=str(baseline_run["id"]),
    )


def record_observation_evidence(
    workspace: Workspace,
    *,
    subject_type: str,
    subject_key: str,
    outcome: str,
    summary: str,
    run_name: str | None = None,
    submission_name: str | None = None,
    strength: int = 0,
    context: dict[str, object] | None = None,
) -> dict[str, object]:
    _validate_subject(subject_type, subject_key)
    if outcome not in _ALLOWED_OUTCOMES:
        raise MemoryError(f"unsupported evidence outcome: {outcome}")
    if not 0 <= strength <= 3:
        raise MemoryError("competition-local evidence strength must be between 0 and 3")
    if run_name is None and submission_name is None:
        raise MemoryError("observation evidence requires --run or --submission provenance")

    source_run_id: str | None = None
    source_experiment_id: str | None = None
    source_submission_id: str | None = None
    domain: str | None = None

    if run_name is not None:
        run = get_run(workspace.db_path, workspace.competition_id, run_name)
        if run is None:
            raise MemoryError(f"run not found: {run_name}")
        if run["status"] != "verified":
            raise MemoryError(f"RUN_NOT_VERIFIED: {run_name}")
        experiment = get_experiment(
            workspace.db_path,
            workspace.competition_id,
            str(run["experiment_name"]),
        )
        assert experiment is not None
        source_run_id = str(run["id"])
        source_experiment_id = str(experiment["id"])
        domain = str(experiment["comparison_domain_hash"])

    if submission_name is not None:
        submission = get_submission(
            workspace.db_path,
            workspace.competition_id,
            submission_name,
        )
        if submission is None:
            raise MemoryError(f"submission not found: {submission_name}")
        source_submission_id = str(submission["id"])
        if source_run_id is None:
            run = get_run(
                workspace.db_path,
                workspace.competition_id,
                str(submission["source_run_name"]),
            )
            assert run is not None
            experiment = get_experiment(
                workspace.db_path,
                workspace.competition_id,
                str(run["experiment_name"]),
            )
            assert experiment is not None
            source_run_id = str(run["id"])
            source_experiment_id = str(experiment["id"])
            domain = str(experiment["comparison_domain_hash"])

    payload = {"source": "observation", **(context or {})}
    return create_evidence_record(
        workspace.db_path,
        competition_id=workspace.competition_id,
        subject_type=subject_type,
        subject_key=subject_key,
        outcome=outcome,
        effect=None,
        strength=strength,
        summary=summary,
        context=payload,
        validation_domain_hash=domain,
        source_experiment_id=source_experiment_id,
        source_run_id=source_run_id,
        source_submission_id=source_submission_id,
    )


def evidence_summaries(workspace: Workspace) -> list[dict[str, object]]:
    rows = list_evidence(workspace.db_path, workspace.competition_id)
    return [
        {
            "id": row["name"],
            "subject": f"{row['subject_type']}:{row['subject_key']}",
            "outcome": row["outcome"],
            "effect": row["effect"],
            "strength": row["strength"],
            "summary": row["summary"],
            "created_at": row["created_at"],
        }
        for row in rows
    ]


def create_finding(
    workspace: Workspace,
    *,
    subject_type: str,
    subject_key: str,
    conclusion: str,
    summary: str,
    evidence_names: list[str],
    contradicting_evidence_names: list[str] | None = None,
    confidence: str = "low",
) -> dict[str, object]:
    _validate_subject(subject_type, subject_key)
    if conclusion not in _ALLOWED_CONCLUSIONS:
        raise MemoryError(f"unsupported finding conclusion: {conclusion}")
    if confidence not in {"low", "medium"}:
        raise MemoryError("competition-local finding confidence is limited to low or medium")
    links = [(name, "supporting") for name in evidence_names]
    links.extend((name, "contradicting") for name in (contradicting_evidence_names or []))
    try:
        return create_finding_record(
            workspace.db_path,
            competition_id=workspace.competition_id,
            subject_type=subject_type,
            subject_key=subject_key,
            conclusion=conclusion,
            summary=summary,
            confidence=confidence,
            evidence_links=links,
        )
    except ValueError as exc:
        raise MemoryError(str(exc)) from exc


def approve_finding(workspace: Workspace, name: str) -> dict[str, object]:
    try:
        return approve_finding_record(workspace.db_path, workspace.competition_id, name)
    except ValueError as exc:
        raise MemoryError(str(exc)) from exc


def finding_detail(workspace: Workspace, name: str) -> dict[str, object]:
    row = get_finding(workspace.db_path, workspace.competition_id, name)
    if row is None:
        raise MemoryError(f"finding not found: {name}")
    row["evidence"] = [
        {
            **item,
            "context": _decode_json(item["context_json"]),
        }
        for item in row["evidence"]
    ]
    return row


def finding_summaries(workspace: Workspace) -> list[dict[str, object]]:
    return [
        {
            "id": row["name"],
            "subject": f"{row['subject_type']}:{row['subject_key']}",
            "conclusion": row["conclusion"],
            "confidence": row["confidence"],
            "status": row["status"],
            "summary": row["summary"],
        }
        for row in list_findings(workspace.db_path, workspace.competition_id)
    ]


def _finding_strength(workspace: Workspace, finding_name: str) -> int:
    finding = get_finding(workspace.db_path, workspace.competition_id, finding_name)
    if finding is None:
        raise MemoryError(f"finding not found: {finding_name}")
    values = [int(item["strength"]) for item in finding["evidence"]]
    return max(values, default=0)


def _conclusion_bucket(conclusion: str) -> str:
    if conclusion == "supported":
        return "positive"
    if conclusion == "rejected":
        return "negative"
    return "neutral"


def _aggregate_applicability(evidence: list[dict[str, object]]) -> dict[str, object]:
    task_types: set[str] = set()
    metric_names: set[str] = set()
    for item in evidence:
        fingerprint = item["fingerprint"]
        if not isinstance(fingerprint, dict):
            continue
        task = fingerprint.get("task")
        metric = fingerprint.get("metric")
        if isinstance(task, dict) and task.get("type"):
            task_types.add(str(task["type"]))
        if isinstance(metric, dict) and metric.get("name"):
            metric_names.add(str(metric["name"]))
    return {
        "task_types": sorted(task_types),
        "metric_names": sorted(metric_names),
        "fingerprint_count": len(evidence),
    }


def learn(workspace: Workspace, finding_name: str | None = None) -> list[dict[str, object]]:
    fingerprint_row = latest_fingerprint(workspace.db_path, workspace.competition_id)
    if fingerprint_row is None:
        raise MemoryError("FINGERPRINT_REQUIRED: set a competition fingerprint before learning")
    fingerprint = _decode_json(fingerprint_row["fingerprint_json"])
    assert isinstance(fingerprint, dict)
    config = load_arena_config(workspace)

    if finding_name is None:
        candidates = approved_findings(workspace.db_path, workspace.competition_id)
    else:
        row = get_finding(workspace.db_path, workspace.competition_id, finding_name)
        if row is None:
            raise MemoryError(f"finding not found: {finding_name}")
        if row["status"] != "approved":
            raise MemoryError(f"FINDING_NOT_APPROVED: {finding_name}")
        candidates = [row]
    if not candidates:
        raise MemoryError("no approved findings available to learn")

    global_path = initialize_knowledge_database(knowledge_db_path())
    results: list[dict[str, object]] = []
    for finding in candidates:
        kind = str(finding["subject_type"])
        key = str(finding["subject_key"])
        previous = latest_knowledge_item(global_path, kind, key)
        prior_evidence = (
            knowledge_evidence(global_path, str(previous["id"])) if previous is not None else []
        )
        if any(
            item["source_competition_id"] == workspace.competition_id
            and item["source_finding_id"] == finding["id"]
            for item in prior_evidence
        ):
            assert previous is not None
            results.append(previous)
            continue

        combined: list[dict[str, object]] = []
        for item in prior_evidence:
            combined.append(
                {
                    "source_competition_id": item["source_competition_id"],
                    "source_competition_slug": item["source_competition_slug"],
                    "source_finding_id": item["source_finding_id"],
                    "source_finding_name": item["source_finding_name"],
                    "conclusion": item["conclusion"],
                    "strength": item["strength"],
                    "fingerprint": _decode_json(item["fingerprint_json"]),
                }
            )
        combined.append(
            {
                "source_competition_id": workspace.competition_id,
                "source_competition_slug": config.competition.slug,
                "source_finding_id": finding["id"],
                "source_finding_name": finding["name"],
                "conclusion": finding["conclusion"],
                "strength": _finding_strength(workspace, str(finding["name"])),
                "fingerprint": fingerprint,
            }
        )

        buckets = [_conclusion_bucket(str(item["conclusion"])) for item in combined]
        positive = buckets.count("positive")
        negative = buckets.count("negative")
        neutral = buckets.count("neutral")
        directional = positive + negative
        consistency = max(positive, negative) / directional if directional else None
        independent = len({str(item["source_competition_id"]) for item in combined})
        confidence = "medium" if independent >= 2 and (consistency or 0.0) >= 0.6 else "low"
        summary = (
            f"Across {independent} competition(s): {positive} supported, "
            f"{negative} rejected, {neutral} mixed/inconclusive."
        )
        result = create_knowledge_version(
            global_path,
            kind=kind,
            key=key,
            title=key.replace("_", " ").title(),
            summary=summary,
            applicability=_aggregate_applicability(combined),
            confidence=confidence,
            independent_competitions=independent,
            positive_count=positive,
            neutral_count=neutral,
            negative_count=negative,
            directional_consistency=consistency,
            evidence=combined,
            supersedes_id=str(previous["id"]) if previous is not None else None,
        )
        results.append(result)
    return results


def _flatten(value: Any, prefix: str = "") -> dict[str, str]:
    result: dict[str, str] = {}
    if isinstance(value, dict):
        for key, item in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(item, dict):
                result.update(_flatten(item, name))
            elif isinstance(item, list):
                result[name] = json.dumps(item, sort_keys=True)
            else:
                result[name] = str(item)
    return result


def _fingerprint_similarity(left: dict[str, object], right: dict[str, object]) -> float:
    a = _flatten(left)
    b = _flatten(right)
    keys = set(a) & set(b)
    if not keys:
        return 0.0
    return sum(1 for key in keys if a[key] == b[key]) / len(keys)


def retrieve_knowledge(
    workspace: Workspace,
    *,
    query: str | None = None,
    limit: int = 10,
) -> list[dict[str, object]]:
    fingerprint_row = latest_fingerprint(workspace.db_path, workspace.competition_id)
    if fingerprint_row is None:
        raise MemoryError("FINGERPRINT_REQUIRED: set a competition fingerprint before retrieval")
    current = _decode_json(fingerprint_row["fingerprint_json"])
    assert isinstance(current, dict)
    path = initialize_knowledge_database(knowledge_db_path())
    rows = list_latest_knowledge(path)
    scored: list[dict[str, object]] = []
    query_lower = query.lower().strip() if query else None

    for row in rows:
        evidence = knowledge_evidence(path, str(row["id"]))
        similarities = []
        max_strength = 0
        contradictions = 0
        for item in evidence:
            fp = _decode_json(item["fingerprint_json"])
            if isinstance(fp, dict):
                similarities.append(_fingerprint_similarity(current, fp))
            max_strength = max(max_strength, int(item["strength"]))
            if item["conclusion"] == "rejected":
                contradictions += 1
        similarity = max(similarities, default=0.0)
        confidence_weight = {"low": 0.3, "medium": 0.6, "high": 0.9}.get(
            str(row["confidence"]), 0.0
        )
        score = similarity * 0.7 + confidence_weight * 0.2 + (max_strength / 4.0) * 0.1
        haystack = f"{row['kind']} {row['key']} {row['title']} {row['summary']}".lower()
        if query_lower:
            if query_lower not in haystack:
                continue
            score += 0.25
        scored.append(
            {
                "kind": row["kind"],
                "key": row["key"],
                "version": row["version"],
                "summary": row["summary"],
                "confidence": row["confidence"],
                "independent_competitions": row["independent_competitions"],
                "positive": row["positive_count"],
                "neutral": row["neutral_count"],
                "negative": row["negative_count"],
                "contradictory_evidence": contradictions,
                "fingerprint_similarity": round(similarity, 6),
                "relevance_score": round(score, 6),
            }
        )
    scored.sort(key=lambda item: (-float(item["relevance_score"]), str(item["kind"]), str(item["key"])))
    return scored[: max(1, limit)]


def knowledge_detail(kind: str, key: str) -> dict[str, object]:
    path = initialize_knowledge_database(knowledge_db_path())
    row = latest_knowledge_item(path, kind, key)
    if row is None:
        raise MemoryError(f"knowledge not found: {kind}:{key}")
    return {
        **row,
        "applicability": _decode_json(row["applicability_json"]),
        "evidence": [
            {
                **item,
                "fingerprint": _decode_json(item["fingerprint_json"]),
            }
            for item in knowledge_evidence(path, str(row["id"]))
        ],
    }


def failure_modes() -> list[dict[str, str]]:
    return [{"key": key, "description": value} for key, value in sorted(FAILURE_MODES.items())]
