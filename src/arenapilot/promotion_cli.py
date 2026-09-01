from __future__ import annotations

import json

import typer

from .memory import MemoryError
from .memory_cli import knowledge_app
from .promotion import (
    approve_knowledge,
    assess_knowledge,
    competition_independence_detail,
    deprecate_knowledge,
    deprecate_technique,
    knowledge_history,
    ranked_knowledge,
    register_technique,
    set_competition_independence,
    technique_detail,
    technique_summaries,
)
from .workspace import WorkspaceError, discover_workspace


independence_app = typer.Typer(
    no_args_is_help=True,
    help="Declare competition evidence independence and lineage groups.",
)
technique_app = typer.Typer(
    no_args_is_help=True,
    help="Manage the global technique registry used by knowledge promotion.",
)


def _fail(exc: Exception, json_output: bool) -> None:
    message = str(exc)
    code = message.split(":", 1)[0] if ":" in message else "PROMOTION_FAILED"
    if json_output:
        typer.echo(json.dumps({"ok": False, "error": {"code": code, "message": message}}))
    else:
        typer.echo(f"ArenaPilot failed: {message}", err=True)
    raise typer.Exit(code=1)


def _subject(value: str) -> tuple[str, str]:
    if ":" not in value:
        raise MemoryError("subject must use <type>:<key>, e.g. technique:frequency_encoding")
    kind, key = value.split(":", 1)
    if not kind.strip() or not key.strip():
        raise MemoryError("subject must contain non-empty type and key")
    return kind.strip(), key.strip()


@independence_app.command("set")
def independence_set_command(
    group: str = typer.Option(..., "--group", help="Independent evidence unit key."),
    dataset_key: str | None = typer.Option(None, "--dataset-key", help="Stable dataset lineage key when known."),
    relation: str = typer.Option(
        "independent",
        "--relation",
        help="independent|derived|same_dataset|related",
    ),
    parent: str | None = typer.Option(None, "--parent", help="Parent competition slug for derived evidence."),
    source: str = typer.Option("manual", "--source", help="Provenance label for this declaration."),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    try:
        workspace = discover_workspace()
        row = set_competition_independence(
            workspace,
            independence_key=group,
            dataset_key=dataset_key,
            relation=relation,
            parent_competition_slug=parent,
            source=source,
        )
    except (MemoryError, WorkspaceError, ValueError, OSError, RuntimeError) as exc:
        _fail(exc, json_output)
    payload = {"ok": True, "independence": row}
    if json_output:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
    else:
        typer.echo(
            f"Independence group: {row['independence_key']}  relation={row['relation']}"
        )


@independence_app.command("show")
def independence_show_command(
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    try:
        row = competition_independence_detail(discover_workspace())
    except (MemoryError, WorkspaceError, ValueError, OSError, RuntimeError) as exc:
        _fail(exc, json_output)
    if json_output:
        typer.echo(json.dumps({"ok": True, "independence": row}, indent=2, sort_keys=True))
    else:
        typer.echo(
            f"{row['independence_key']}  relation={row['relation']}  "
            f"dataset={row.get('dataset_key') or '-'}"
        )


@technique_app.command("register")
def technique_register_command(
    key: str = typer.Argument(..., help="Stable technique key, e.g. frequency_encoding."),
    title: str | None = typer.Option(None, "--title"),
    category: str = typer.Option("general", "--category"),
    description: str = typer.Option("", "--description"),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    try:
        row = register_technique(
            key,
            title=title,
            category=category,
            description=description,
        )
    except (MemoryError, ValueError, OSError, RuntimeError) as exc:
        _fail(exc, json_output)
    if json_output:
        typer.echo(json.dumps({"ok": True, "technique": row}, indent=2, sort_keys=True))
    else:
        typer.echo(f"Registered technique {row['key']} ({row['category']}).")


@technique_app.command("list")
def technique_list_command(
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    try:
        rows = technique_summaries()
    except (MemoryError, ValueError, OSError, RuntimeError) as exc:
        _fail(exc, json_output)
    if json_output:
        typer.echo(json.dumps({"ok": True, "techniques": rows}, indent=2, sort_keys=True))
    else:
        for row in rows:
            typer.echo(f"{row['key']}  {row['status']}  {row['category']}  {row['title']}")


@technique_app.command("show")
def technique_show_command(
    key: str = typer.Argument(...),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    try:
        row = technique_detail(key)
    except (MemoryError, ValueError, OSError, RuntimeError) as exc:
        _fail(exc, json_output)
    if json_output:
        typer.echo(json.dumps({"ok": True, "technique": row}, indent=2, sort_keys=True))
    else:
        typer.echo(f"{row['key']}  {row['status']}  {row['category']}")
        typer.echo(str(row["description"]))


@technique_app.command("deprecate")
def technique_deprecate_command(
    key: str = typer.Argument(...),
    reason: str = typer.Option(..., "--reason"),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    try:
        row = deprecate_technique(key, reason)
    except (MemoryError, ValueError, OSError, RuntimeError) as exc:
        _fail(exc, json_output)
    if json_output:
        typer.echo(json.dumps({"ok": True, "technique": row}, indent=2, sort_keys=True))
    else:
        typer.echo(f"Deprecated technique {key}.")


@knowledge_app.command("assess")
def knowledge_assess_command(
    subject: str = typer.Argument(..., help="type:key knowledge subject."),
    version: int | None = typer.Option(None, "--version", min=1),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    try:
        kind, key = _subject(subject)
        row = assess_knowledge(kind, key, version=version)
    except (MemoryError, ValueError, OSError, RuntimeError) as exc:
        _fail(exc, json_output)
    if json_output:
        typer.echo(json.dumps({"ok": True, "assessment": row}, indent=2, sort_keys=True))
    else:
        typer.echo(
            f"{kind}:{key} v{row['version']}  raw={row['raw_competitions']}  "
            f"independent={row['independent_units']}  consistency={row['directional_consistency']}"
        )
        typer.echo(
            f"effective={row['effective_confidence']}  major_contradiction={row['major_contradiction']}"
        )


@knowledge_app.command("approve")
def knowledge_approve_command(
    subject: str = typer.Argument(..., help="type:key knowledge subject."),
    confidence: str = typer.Option(..., "--confidence", help="low|medium|high"),
    reason: str | None = typer.Option(None, "--reason"),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    try:
        kind, key = _subject(subject)
        row = approve_knowledge(kind, key, confidence=confidence, reason=reason)
    except (MemoryError, ValueError, OSError, RuntimeError) as exc:
        _fail(exc, json_output)
    result = {
        "ok": True,
        "subject": subject,
        "version": row["knowledge"]["version"],
        "status": row["promotion"]["status"],
        "confidence": row["promotion"]["approved_confidence"],
        "assessment": row["assessment"],
    }
    if json_output:
        typer.echo(json.dumps(result, indent=2, sort_keys=True))
    else:
        typer.echo(
            f"Approved {subject} v{result['version']} as {result['confidence']} confidence."
        )


@knowledge_app.command("deprecate")
def knowledge_deprecate_command(
    subject: str = typer.Argument(..., help="type:key knowledge subject."),
    reason: str = typer.Option(..., "--reason"),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    try:
        kind, key = _subject(subject)
        row = deprecate_knowledge(kind, key, reason)
    except (MemoryError, ValueError, OSError, RuntimeError) as exc:
        _fail(exc, json_output)
    result = {
        "ok": True,
        "subject": subject,
        "version": row["knowledge"]["version"],
        "status": row["promotion"]["status"],
        "reason": row["promotion"]["reason"],
    }
    if json_output:
        typer.echo(json.dumps(result, indent=2, sort_keys=True))
    else:
        typer.echo(f"Deprecated {subject} v{result['version']}.")


@knowledge_app.command("history")
def knowledge_history_command(
    subject: str = typer.Argument(..., help="type:key knowledge subject."),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    try:
        kind, key = _subject(subject)
        rows = knowledge_history(kind, key)
    except (MemoryError, ValueError, OSError, RuntimeError) as exc:
        _fail(exc, json_output)
    if json_output:
        typer.echo(json.dumps({"ok": True, "history": rows}, indent=2, sort_keys=True))
    else:
        for row in rows:
            typer.echo(
                f"v{row['version']}  promotion={row.get('promotion_status') or 'candidate'}  "
                f"effective={row.get('effective_confidence') or row['confidence']}"
            )


@knowledge_app.command("ranked")
def knowledge_ranked_command(
    query: str | None = typer.Option(None, "--query"),
    limit: int = typer.Option(10, "--limit", min=1, max=100),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    try:
        rows = ranked_knowledge(discover_workspace(), query=query, limit=limit)
    except (MemoryError, WorkspaceError, ValueError, OSError, RuntimeError) as exc:
        _fail(exc, json_output)
    if json_output:
        typer.echo(json.dumps({"ok": True, "knowledge": rows}, indent=2, sort_keys=True))
    else:
        for row in rows:
            typer.echo(
                f"{row['kind']}:{row['key']}  score={row['relevance_score']}  "
                f"status={row['status']}  confidence={row['confidence']}  "
                f"independent={row['independent_units']}"
            )
