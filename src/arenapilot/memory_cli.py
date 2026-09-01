from __future__ import annotations

import json

import typer

from .memory import (
    MemoryError,
    approve_finding,
    create_finding,
    evidence_summaries,
    failure_modes,
    finding_detail,
    finding_summaries,
    fingerprint_summary,
    knowledge_detail,
    learn,
    record_comparison_evidence,
    record_observation_evidence,
    retrieve_knowledge,
    set_competition_fingerprint,
)
from .memory_schema import initialize_workspace_memory_schema
from .workspace import WorkspaceError, discover_workspace


fingerprint_app = typer.Typer(no_args_is_help=True, help="Record competition fingerprints.")
evidence_app = typer.Typer(no_args_is_help=True, help="Record immutable experimental evidence.")
finding_app = typer.Typer(no_args_is_help=True, help="Create and approve competition-local findings.")
knowledge_app = typer.Typer(no_args_is_help=True, help="Retrieve cross-competition knowledge candidates.")
failure_app = typer.Typer(no_args_is_help=True, help="Inspect the built-in failure-mode registry.")


def _fail(exc: Exception, json_output: bool) -> None:
    message = str(exc)
    code = message.split(":", 1)[0] if ":" in message else "MEMORY_FAILED"
    if json_output:
        typer.echo(json.dumps({"ok": False, "error": {"code": code, "message": message}}))
    else:
        typer.echo(f"ArenaPilot failed: {message}", err=True)
    raise typer.Exit(code=1)


def _workspace():
    workspace = discover_workspace()
    initialize_workspace_memory_schema(workspace.db_path)
    return workspace


def _json_object(raw: str | None, label: str) -> dict[str, object]:
    if raw is None:
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise MemoryError(f"{label} must be valid JSON") from exc
    if not isinstance(value, dict):
        raise MemoryError(f"{label} must be a JSON object")
    return value


def _subject(value: str) -> tuple[str, str]:
    if ":" not in value:
        raise MemoryError("subject must use <type>:<key>, e.g. technique:frequency_encoding")
    subject_type, subject_key = value.split(":", 1)
    return subject_type.strip(), subject_key.strip()


@fingerprint_app.command("set")
def fingerprint_set_command(
    observed_json: str | None = typer.Option(
        None,
        "--observed-json",
        help="Observed dataset facts as a JSON object.",
    ),
    inferred_json: str | None = typer.Option(
        None,
        "--inferred-json",
        help="Inferred labels/properties as a JSON object.",
    ),
    source: str = typer.Option("manual", "--source", help="Fingerprint provenance label."),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    try:
        workspace = _workspace()
        row = set_competition_fingerprint(
            workspace,
            observed=_json_object(observed_json, "observed-json"),
            inferred=_json_object(inferred_json, "inferred-json"),
            source=source,
        )
        payload = fingerprint_summary(workspace)
    except (MemoryError, WorkspaceError, ValueError, OSError, RuntimeError) as exc:
        _fail(exc, json_output)
    assert payload is not None
    result = {
        "ok": True,
        "fingerprint_hash": row["fingerprint_hash"],
        "source": row["source"],
        "fingerprint": payload["fingerprint"],
    }
    if json_output:
        typer.echo(json.dumps(result, indent=2, sort_keys=True))
    else:
        typer.echo(f"Fingerprint: {str(row['fingerprint_hash'])[:12]}... ({row['source']})")


@fingerprint_app.command("show")
def fingerprint_show_command(
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    try:
        workspace = _workspace()
        payload = fingerprint_summary(workspace)
        if payload is None:
            raise MemoryError("fingerprint not found")
    except (MemoryError, WorkspaceError, ValueError, OSError, RuntimeError) as exc:
        _fail(exc, json_output)
    result = {"ok": True, **payload}
    if json_output:
        typer.echo(json.dumps(result, indent=2, sort_keys=True))
    else:
        typer.echo(json.dumps(payload["fingerprint"], indent=2, sort_keys=True))


@evidence_app.command("compare")
def evidence_compare_command(
    subject: str = typer.Option(..., "--subject", help="type:key memory subject."),
    baseline: str = typer.Option(..., "--baseline", help="Baseline Experiment ID."),
    candidate: str = typer.Option(..., "--candidate", help="Candidate Experiment ID."),
    summary: str = typer.Option(..., "--summary", help="Interpretation-free evidence summary."),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    try:
        workspace = _workspace()
        subject_type, subject_key = _subject(subject)
        row = record_comparison_evidence(
            workspace,
            subject_type=subject_type,
            subject_key=subject_key,
            baseline=baseline,
            candidate=candidate,
            summary=summary,
        )
    except (MemoryError, WorkspaceError, ValueError, OSError, RuntimeError) as exc:
        _fail(exc, json_output)
    result = {
        "ok": True,
        "evidence": row["name"],
        "subject": f"{row['subject_type']}:{row['subject_key']}",
        "outcome": row["outcome"],
        "effect": row["effect"],
        "strength": row["strength"],
    }
    if json_output:
        typer.echo(json.dumps(result, indent=2, sort_keys=True))
    else:
        typer.echo(
            f"{row['name']}  {row['outcome']}  effect={row['effect']}  "
            f"strength={row['strength']}"
        )


@evidence_app.command("note")
def evidence_note_command(
    subject: str = typer.Option(..., "--subject", help="type:key memory subject."),
    outcome: str = typer.Option(..., "--outcome", help="positive|neutral|negative|warning"),
    summary: str = typer.Option(..., "--summary", help="Observed fact or failure summary."),
    run: str | None = typer.Option(None, "--run", help="VERIFIED Run provenance."),
    submission: str | None = typer.Option(None, "--submission", help="Submission provenance."),
    strength: int = typer.Option(0, "--strength", min=0, max=3),
    context_json: str | None = typer.Option(None, "--context-json", help="Additional context JSON object."),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    try:
        workspace = _workspace()
        subject_type, subject_key = _subject(subject)
        row = record_observation_evidence(
            workspace,
            subject_type=subject_type,
            subject_key=subject_key,
            outcome=outcome,
            summary=summary,
            run_name=run,
            submission_name=submission,
            strength=strength,
            context=_json_object(context_json, "context-json"),
        )
    except (MemoryError, WorkspaceError, ValueError, OSError, RuntimeError) as exc:
        _fail(exc, json_output)
    result = {
        "ok": True,
        "evidence": row["name"],
        "subject": f"{row['subject_type']}:{row['subject_key']}",
        "outcome": row["outcome"],
        "strength": row["strength"],
    }
    if json_output:
        typer.echo(json.dumps(result, indent=2, sort_keys=True))
    else:
        typer.echo(f"{row['name']}  {row['outcome']}  {row['summary']}")


@evidence_app.command("list")
def evidence_list_command(
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    try:
        rows = evidence_summaries(_workspace())
    except (MemoryError, WorkspaceError, ValueError, OSError, RuntimeError) as exc:
        _fail(exc, json_output)
    if json_output:
        typer.echo(json.dumps({"ok": True, "evidence": rows}, indent=2, sort_keys=True))
    else:
        for row in rows:
            typer.echo(f"{row['id']}  {row['subject']}  {row['outcome']}  {row['summary']}")


@finding_app.command("create")
def finding_create_command(
    subject: str = typer.Option(..., "--subject", help="type:key memory subject."),
    conclusion: str = typer.Option(..., "--conclusion", help="supported|rejected|mixed|inconclusive"),
    summary: str = typer.Option(..., "--summary", help="Competition-local interpretation."),
    evidence: list[str] = typer.Option(..., "--evidence", help="Supporting evidence ID. Repeatable."),
    contradicting: list[str] = typer.Option([], "--contradicting", help="Contradicting evidence ID. Repeatable."),
    confidence: str = typer.Option("low", "--confidence", help="low|medium"),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    try:
        workspace = _workspace()
        subject_type, subject_key = _subject(subject)
        row = create_finding(
            workspace,
            subject_type=subject_type,
            subject_key=subject_key,
            conclusion=conclusion,
            summary=summary,
            evidence_names=evidence,
            contradicting_evidence_names=contradicting,
            confidence=confidence,
        )
    except (MemoryError, WorkspaceError, ValueError, OSError, RuntimeError) as exc:
        _fail(exc, json_output)
    result = {
        "ok": True,
        "finding": row["name"],
        "status": row["status"],
        "subject": f"{row['subject_type']}:{row['subject_key']}",
        "conclusion": row["conclusion"],
        "confidence": row["confidence"],
    }
    if json_output:
        typer.echo(json.dumps(result, indent=2, sort_keys=True))
    else:
        typer.echo(f"Created {row['name']} ({row['conclusion']}, {row['confidence']}).")


@finding_app.command("approve")
def finding_approve_command(
    name: str = typer.Argument(..., help="Finding ID, e.g. finding001."),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    try:
        row = approve_finding(_workspace(), name)
    except (MemoryError, WorkspaceError, ValueError, OSError, RuntimeError) as exc:
        _fail(exc, json_output)
    result = {"ok": True, "finding": row["name"], "status": row["status"]}
    if json_output:
        typer.echo(json.dumps(result, indent=2, sort_keys=True))
    else:
        typer.echo(f"Approved {name}.")


@finding_app.command("show")
def finding_show_command(
    name: str = typer.Argument(..., help="Finding ID, e.g. finding001."),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    try:
        row = finding_detail(_workspace(), name)
    except (MemoryError, WorkspaceError, ValueError, OSError, RuntimeError) as exc:
        _fail(exc, json_output)
    if json_output:
        typer.echo(json.dumps({"ok": True, "finding": row}, indent=2, sort_keys=True))
    else:
        typer.echo(f"{row['name']}  {row['subject_type']}:{row['subject_key']}")
        typer.echo(f"{row['conclusion']} / {row['confidence']} / {row['status']}")
        typer.echo(str(row["summary"]))
        for item in row["evidence"]:
            typer.echo(f"- {item['name']} [{item['role']}] {item['outcome']}: {item['summary']}")


@finding_app.command("list")
def finding_list_command(
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    try:
        rows = finding_summaries(_workspace())
    except (MemoryError, WorkspaceError, ValueError, OSError, RuntimeError) as exc:
        _fail(exc, json_output)
    if json_output:
        typer.echo(json.dumps({"ok": True, "findings": rows}, indent=2, sort_keys=True))
    else:
        for row in rows:
            typer.echo(
                f"{row['id']}  {row['status']}  {row['subject']}  "
                f"{row['conclusion']}  {row['confidence']}"
            )


def learn_command(
    finding: str | None = typer.Option(None, "--finding", help="Promote one approved finding; default all."),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    try:
        rows = learn(_workspace(), finding_name=finding)
    except (MemoryError, WorkspaceError, ValueError, OSError, RuntimeError) as exc:
        _fail(exc, json_output)
    result = {
        "ok": True,
        "knowledge": [
            {
                "kind": row["kind"],
                "key": row["key"],
                "version": row["version"],
                "confidence": row["confidence"],
                "independent_competitions": row["independent_competitions"],
            }
            for row in rows
        ],
    }
    if json_output:
        typer.echo(json.dumps(result, indent=2, sort_keys=True))
    else:
        for row in result["knowledge"]:
            typer.echo(
                f"{row['kind']}:{row['key']} v{row['version']}  "
                f"{row['confidence']}  competitions={row['independent_competitions']}"
            )


@knowledge_app.command("retrieve")
def knowledge_retrieve_command(
    query: str | None = typer.Option(None, "--query", help="Optional lexical filter."),
    limit: int = typer.Option(10, "--limit", min=1, max=100),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    try:
        rows = retrieve_knowledge(_workspace(), query=query, limit=limit)
    except (MemoryError, WorkspaceError, ValueError, OSError, RuntimeError) as exc:
        _fail(exc, json_output)
    if json_output:
        typer.echo(json.dumps({"ok": True, "knowledge": rows}, indent=2, sort_keys=True))
    else:
        for row in rows:
            typer.echo(
                f"{row['kind']}:{row['key']}  score={row['relevance_score']}  "
                f"confidence={row['confidence']}  competitions={row['independent_competitions']}  "
                f"contradictions={row['contradictory_evidence']}"
            )


@knowledge_app.command("show")
def knowledge_show_command(
    subject: str = typer.Argument(..., help="type:key knowledge subject."),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    try:
        kind, key = _subject(subject)
        row = knowledge_detail(kind, key)
    except (MemoryError, WorkspaceError, ValueError, OSError, RuntimeError) as exc:
        _fail(exc, json_output)
    if json_output:
        typer.echo(json.dumps({"ok": True, "knowledge": row}, indent=2, sort_keys=True))
    else:
        typer.echo(f"{row['kind']}:{row['key']} v{row['version']}  {row['confidence']}")
        typer.echo(str(row["summary"]))
        typer.echo(f"Independent competitions: {row['independent_competitions']}")
        for item in row["evidence"]:
            typer.echo(
                f"- {item['source_competition_slug']} / {item['source_finding_name']}: "
                f"{item['conclusion']}"
            )


@failure_app.command("list")
def failure_list_command(
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    rows = failure_modes()
    if json_output:
        typer.echo(json.dumps({"ok": True, "failure_modes": rows}, indent=2, sort_keys=True))
    else:
        for row in rows:
            typer.echo(f"{row['key']}  {row['description']}")
