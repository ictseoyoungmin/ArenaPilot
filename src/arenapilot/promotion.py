from __future__ import annotations

import json
from collections import defaultdict
from typing import Any

from .memory import MemoryError
from .memory_store import (
    knowledge_db_path,
    latest_fingerprint,
)
from .promotion_store import (
    approve_knowledge_record,
    deprecate_knowledge_record,
    deprecate_technique_record,
    get_promotion,
    get_technique_record,
    get_workspace_independence_profile,
    independence_registry,
    initialize_promotion_database,
    knowledge_evidence_rows,
    knowledge_history_rows,
    knowledge_version,
    list_technique_records,
    register_technique_record,
    retrieval_rows,
    save_assessment,
    set_independence_profile,
)
from .workspace import Workspace, load_arena_config


_ALLOWED_RELATIONS = {"independent", "derived", "same_dataset", "related"}
_ALLOWED_APPROVAL_CONFIDENCE = {"low", "medium", "high"}


def _decode_json(value: object) -> object:
    if isinstance(value, str):
        return json.loads(value)
    return value


def _flatten(value: Any, prefix: str = "") -> dict[str, str]:
    result: dict[str, str] = {}
    if isinstance(value, dict):
        for key, item in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(item, dict):
                result.update(_flatten(item, name))
            elif isinstance(item, list):
                result[name] = json.dumps(item, sort_keys=True, ensure_ascii=False)
            else:
                result[name] = str(item)
    return result


def _similarity(left: object, right: object) -> float:
    if not isinstance(left, dict) or not isinstance(right, dict):
        return 0.0
    a = _flatten(left)
    b = _flatten(right)
    keys = set(a) & set(b)
    if not keys:
        return 0.0
    return sum(1 for key in keys if a[key] == b[key]) / len(keys)


def set_competition_independence(
    workspace: Workspace,
    *,
    independence_key: str,
    dataset_key: str | None = None,
    relation: str = "independent",
    parent_competition_slug: str | None = None,
    source: str = "manual",
) -> dict[str, object]:
    key = independence_key.strip()
    if not key:
        raise MemoryError("independence group cannot be empty")
    if relation not in _ALLOWED_RELATIONS:
        raise MemoryError(f"unsupported independence relation: {relation}")
    if relation == "derived" and not parent_competition_slug:
        raise MemoryError("derived independence relation requires --parent")
    if relation == "same_dataset" and not dataset_key:
        raise MemoryError("same_dataset relation requires --dataset-key")

    config = load_arena_config(workspace)
    fingerprint = latest_fingerprint(workspace.db_path, workspace.competition_id)
    fingerprint_hash = str(fingerprint["fingerprint_hash"]) if fingerprint else None
    return set_independence_profile(
        workspace.db_path,
        initialize_promotion_database(knowledge_db_path()),
        competition_id=workspace.competition_id,
        competition_slug=config.competition.slug,
        independence_key=key,
        dataset_key=dataset_key,
        relation=relation,
        parent_competition_slug=parent_competition_slug,
        fingerprint_hash=fingerprint_hash,
        source=source,
    )


def competition_independence_detail(workspace: Workspace) -> dict[str, object]:
    row = get_workspace_independence_profile(workspace.db_path, workspace.competition_id)
    if row is None:
        config = load_arena_config(workspace)
        return {
            "competition_id": workspace.competition_id,
            "competition_slug": config.competition.slug,
            "independence_key": f"competition:{workspace.competition_id}",
            "dataset_key": None,
            "relation": "implicit_independent",
            "parent_competition_slug": None,
            "source": "fallback",
        }
    return row


def register_technique(
    key: str,
    *,
    title: str | None = None,
    category: str = "general",
    description: str = "",
) -> dict[str, object]:
    normalized = key.strip()
    if not normalized:
        raise MemoryError("technique key cannot be empty")
    return register_technique_record(
        initialize_promotion_database(knowledge_db_path()),
        key=normalized,
        title=(title or normalized.replace("_", " ").title()).strip(),
        category=category.strip() or "general",
        description=description.strip(),
    )


def technique_detail(key: str) -> dict[str, object]:
    row = get_technique_record(initialize_promotion_database(knowledge_db_path()), key)
    if row is None:
        raise MemoryError(f"technique not found: {key}")
    return row


def technique_summaries() -> list[dict[str, object]]:
    return list_technique_records(initialize_promotion_database(knowledge_db_path()))


def deprecate_technique(key: str, reason: str) -> dict[str, object]:
    try:
        return deprecate_technique_record(
            initialize_promotion_database(knowledge_db_path()), key, reason
        )
    except ValueError as exc:
        raise MemoryError(str(exc)) from exc


def _bucket(conclusion: str) -> str:
    if conclusion == "supported":
        return "positive"
    if conclusion == "rejected":
        return "negative"
    return "neutral"


def assess_knowledge(
    kind: str,
    key: str,
    *,
    version: int | None = None,
) -> dict[str, object]:
    path = initialize_promotion_database(knowledge_db_path())
    knowledge = knowledge_version(path, kind, key, version)
    if knowledge is None:
        raise MemoryError(f"knowledge not found: {kind}:{key}")
    evidence = knowledge_evidence_rows(path, str(knowledge["id"]))
    registry = independence_registry(path)

    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    raw_competitions: set[str] = set()
    for item in evidence:
        competition_id = str(item["source_competition_id"])
        raw_competitions.add(competition_id)
        profile = registry.get(competition_id)
        group_key = (
            str(profile["independence_key"])
            if profile is not None
            else f"competition:{competition_id}"
        )
        groups[group_key].append(item)

    group_rows: list[dict[str, object]] = []
    for group_key, items in sorted(groups.items()):
        buckets = [_bucket(str(item["conclusion"])) for item in items]
        positive = buckets.count("positive")
        negative = buckets.count("negative")
        if positive > negative:
            outcome = "positive"
        elif negative > positive:
            outcome = "negative"
        else:
            outcome = "neutral"
        group_rows.append(
            {
                "independence_key": group_key,
                "outcome": outcome,
                "source_competitions": sorted(
                    {str(item["source_competition_slug"]) for item in items}
                ),
                "finding_count": len(items),
                "max_strength": max((int(item["strength"]) for item in items), default=0),
            }
        )

    positive_units = sum(1 for item in group_rows if item["outcome"] == "positive")
    negative_units = sum(1 for item in group_rows if item["outcome"] == "negative")
    neutral_units = sum(1 for item in group_rows if item["outcome"] == "neutral")
    directional_units = positive_units + negative_units
    consistency = (
        max(positive_units, negative_units) / directional_units
        if directional_units
        else None
    )
    majority = (
        "positive"
        if positive_units > negative_units
        else "negative"
        if negative_units > positive_units
        else None
    )
    major_contradiction = False
    if majority is not None:
        major_contradiction = any(
            item["outcome"] not in {majority, "neutral"} and int(item["max_strength"]) >= 2
            for item in group_rows
        )

    independent_units = len(group_rows)
    effective_confidence = (
        "medium"
        if independent_units >= 2 and (consistency or 0.0) >= 0.60
        else "low"
    )
    assessment = {
        "knowledge_id": knowledge["id"],
        "kind": knowledge["kind"],
        "key": knowledge["key"],
        "version": knowledge["version"],
        "raw_competitions": len(raw_competitions),
        "independent_units": independent_units,
        "directional_units": directional_units,
        "positive_units": positive_units,
        "neutral_units": neutral_units,
        "negative_units": negative_units,
        "directional_consistency": consistency,
        "major_contradiction": major_contradiction,
        "effective_confidence": effective_confidence,
        "groups": group_rows,
    }
    save_assessment(path, str(knowledge["id"]), assessment)
    return assessment


def _require_technique_registry(kind: str, key: str) -> None:
    if kind != "technique":
        return
    row = get_technique_record(initialize_promotion_database(knowledge_db_path()), key)
    if row is None:
        raise MemoryError(f"TECHNIQUE_NOT_REGISTERED: {key}")
    if row["status"] != "active":
        raise MemoryError(f"TECHNIQUE_DEPRECATED: {key}")


def approve_knowledge(
    kind: str,
    key: str,
    *,
    confidence: str,
    reason: str | None = None,
) -> dict[str, object]:
    if confidence not in _ALLOWED_APPROVAL_CONFIDENCE:
        raise MemoryError(f"unsupported approval confidence: {confidence}")
    path = initialize_promotion_database(knowledge_db_path())
    knowledge = knowledge_version(path, kind, key)
    if knowledge is None:
        raise MemoryError(f"knowledge not found: {kind}:{key}")
    _require_technique_registry(kind, key)
    assessment = assess_knowledge(kind, key, version=int(knowledge["version"]))

    if confidence == "medium":
        if int(assessment["independent_units"]) < 2 or float(
            assessment["directional_consistency"] or 0.0
        ) < 0.60:
            raise MemoryError(
                "KNOWLEDGE_PROMOTION_BLOCKED: medium requires >=2 independent units and >=0.60 consistency"
            )
    elif confidence == "high":
        if int(assessment["directional_units"]) < 4:
            raise MemoryError(
                "KNOWLEDGE_PROMOTION_BLOCKED: high requires >=4 independent directional units"
            )
        if float(assessment["directional_consistency"] or 0.0) < 0.70:
            raise MemoryError(
                "KNOWLEDGE_PROMOTION_BLOCKED: high requires >=0.70 directional consistency"
            )
        if bool(assessment["major_contradiction"]):
            raise MemoryError(
                "KNOWLEDGE_PROMOTION_BLOCKED: high confidence has a major unresolved contradiction"
            )

    promotion = approve_knowledge_record(
        path,
        knowledge=knowledge,
        confidence=confidence,
        reason=reason,
    )
    return {
        "knowledge": knowledge,
        "assessment": assessment,
        "promotion": promotion,
    }


def deprecate_knowledge(kind: str, key: str, reason: str) -> dict[str, object]:
    path = initialize_promotion_database(knowledge_db_path())
    knowledge = knowledge_version(path, kind, key)
    if knowledge is None:
        raise MemoryError(f"knowledge not found: {kind}:{key}")
    try:
        promotion = deprecate_knowledge_record(path, str(knowledge["id"]), reason)
    except ValueError as exc:
        raise MemoryError(str(exc)) from exc
    return {"knowledge": knowledge, "promotion": promotion}


def knowledge_history(kind: str, key: str) -> list[dict[str, object]]:
    path = initialize_promotion_database(knowledge_db_path())
    rows = knowledge_history_rows(path, kind, key)
    if not rows:
        raise MemoryError(f"knowledge not found: {kind}:{key}")
    return rows


def _effective_status(path, knowledge_id: str) -> tuple[str, str]:
    promotion = get_promotion(path, knowledge_id)
    if promotion is None:
        return "candidate", "low"
    return (
        str(promotion["status"]),
        str(promotion["approved_confidence"] or "low"),
    )


def ranked_knowledge(
    workspace: Workspace,
    *,
    query: str | None = None,
    limit: int = 10,
) -> list[dict[str, object]]:
    fingerprint_row = latest_fingerprint(workspace.db_path, workspace.competition_id)
    if fingerprint_row is None:
        raise MemoryError("FINGERPRINT_REQUIRED: set a competition fingerprint before retrieval")
    current = _decode_json(fingerprint_row["fingerprint_json"])
    if not isinstance(current, dict):
        raise MemoryError("invalid competition fingerprint")

    path = initialize_promotion_database(knowledge_db_path())
    rows = retrieval_rows(path)
    query_lower = query.lower().strip() if query else None
    scored: list[dict[str, object]] = []

    for row in rows:
        status, approved_confidence = _effective_status(path, str(row["id"]))
        if status == "deprecated":
            continue
        if row["kind"] == "technique":
            technique = get_technique_record(path, str(row["key"]))
            if technique is not None and technique["status"] == "deprecated":
                continue

        assessment = assess_knowledge(
            str(row["kind"]), str(row["key"]), version=int(row["version"])
        )
        evidence = knowledge_evidence_rows(path, str(row["id"]))
        observed_scores: list[float] = []
        inferred_scores: list[float] = []
        core_scores: list[float] = []
        max_strength = 0
        for item in evidence:
            fp = _decode_json(item["fingerprint_json"])
            if not isinstance(fp, dict):
                continue
            observed_scores.append(_similarity(current.get("observed"), fp.get("observed")))
            inferred_scores.append(_similarity(current.get("inferred"), fp.get("inferred")))
            core_scores.append(
                (_similarity(current.get("task"), fp.get("task")) + _similarity(current.get("metric"), fp.get("metric"))) / 2.0
            )
            max_strength = max(max_strength, int(item["strength"]))

        observed = max(observed_scores, default=0.0)
        inferred = max(inferred_scores, default=0.0)
        core = max(core_scores, default=0.0)
        confidence = approved_confidence if status == "approved" else str(assessment["effective_confidence"])
        confidence_weight = {"low": 0.25, "medium": 0.60, "high": 1.0}.get(confidence, 0.0)
        independence_weight = min(int(assessment["independent_units"]), 4) / 4.0
        directional = max(int(assessment["directional_units"]), 1)
        contradiction_ratio = min(
            int(assessment["positive_units"]), int(assessment["negative_units"])
        ) / directional
        status_bonus = 0.10 if status == "approved" else 0.0
        score = (
            core * 0.20
            + observed * 0.30
            + inferred * 0.10
            + confidence_weight * 0.15
            + independence_weight * 0.15
            + (max_strength / 3.0) * 0.10
            + status_bonus
            - contradiction_ratio * 0.15
        )

        haystack = f"{row['kind']} {row['key']} {row['title']} {row['summary']}".lower()
        if query_lower:
            if query_lower not in haystack:
                continue
            score += 0.20
        scored.append(
            {
                "kind": row["kind"],
                "key": row["key"],
                "version": row["version"],
                "status": status,
                "confidence": confidence,
                "summary": row["summary"],
                "independent_units": assessment["independent_units"],
                "raw_competitions": assessment["raw_competitions"],
                "directional_consistency": assessment["directional_consistency"],
                "major_contradiction": assessment["major_contradiction"],
                "observed_similarity": round(observed, 6),
                "inferred_similarity": round(inferred, 6),
                "core_similarity": round(core, 6),
                "relevance_score": round(score, 6),
            }
        )

    scored.sort(
        key=lambda item: (
            -float(item["relevance_score"]),
            str(item["kind"]),
            str(item["key"]),
        )
    )
    return scored[: max(1, limit)]
