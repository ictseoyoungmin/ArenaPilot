from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import shutil
from pathlib import Path
from typing import Protocol

import yaml

from .kaggle_backend import KaggleBackendError, _run_kaggle_cli
from .models import PredictionType, ValidationSpec
from .runstore import get_run
from .submission_store import (
    create_submission_record,
    get_submission,
    list_submissions,
    mark_submitted,
    mark_validated,
    submission_budget_usage,
    update_submission_score,
)
from .workspace import Workspace, WorkspaceError, load_arena_config


class SubmissionError(WorkspaceError):
    pass


class SubmissionProvider(Protocol):
    def submit_file(self, competition: str, file_path: Path, message: str) -> str: ...

    def submission_status(self, competition: str, submission_ref: str) -> dict[str, object]: ...

    def submission_limits(self, competition: str) -> dict[str, object] | None: ...


class KaggleSubmissionProvider:
    def submit_file(self, competition: str, file_path: Path, message: str) -> str:
        try:
            process = _run_kaggle_cli(
                [
                    "competitions",
                    "submit",
                    competition,
                    "-f",
                    str(file_path),
                    "-m",
                    message,
                ]
            )
        except KaggleBackendError as exc:
            raise SubmissionError(str(exc)) from exc
        output = "\n".join(part for part in (process.stdout, process.stderr) if part)
        match = re.search(r"Submission ref:\s*([0-9]+)", output, flags=re.IGNORECASE)
        if match is None:
            raise SubmissionError(
                "KAGGLE_SUBMISSION_REF_MISSING: Kaggle accepted no parseable submission reference"
            )
        return match.group(1)

    def submission_status(self, competition: str, submission_ref: str) -> dict[str, object]:
        try:
            process = _run_kaggle_cli(
                [
                    "competitions",
                    "submissions",
                    competition,
                    "--format",
                    "json",
                    "--page-size",
                    "100",
                ]
            )
        except KaggleBackendError as exc:
            raise SubmissionError(str(exc)) from exc
        try:
            payload = json.loads(process.stdout)
        except json.JSONDecodeError as exc:
            raise SubmissionError("KAGGLE_SUBMISSIONS_JSON_INVALID") from exc

        rows: list[object]
        if isinstance(payload, list):
            rows = payload
        elif isinstance(payload, dict):
            candidate = payload.get("submissions") or payload.get("items") or payload.get("results")
            rows = candidate if isinstance(candidate, list) else [payload]
        else:
            rows = []

        for item in rows:
            if not isinstance(item, dict):
                continue
            ref = item.get("ref") or item.get("id") or item.get("submissionId")
            if ref is not None and str(ref) == str(submission_ref):
                return dict(item)
        raise SubmissionError(f"KAGGLE_SUBMISSION_NOT_FOUND: {submission_ref}")

    def submission_limits(self, competition: str) -> dict[str, object] | None:
        try:
            process = _run_kaggle_cli(
                ["competitions", "submission-limits", competition, "--json"]
            )
            payload = json.loads(process.stdout)
        except (KaggleBackendError, json.JSONDecodeError):
            return None
        return dict(payload) if isinstance(payload, dict) else None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _submission_dir(workspace: Workspace) -> Path:
    path = workspace.root / "submissions"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _run_dir(workspace: Workspace, run_name: str) -> Path:
    return workspace.root / "outputs" / "runs" / run_name


def create_submission(
    workspace: Workspace,
    *,
    run_name: str,
    file_path: Path | None = None,
    message: str | None = None,
) -> dict[str, object]:
    config = load_arena_config(workspace)
    run = get_run(workspace.db_path, workspace.competition_id, run_name)
    if run is None:
        raise SubmissionError(f"RUN_NOT_FOUND: {run_name}")
    if config.submission.require_verified_run and run["status"] != "verified":
        raise SubmissionError("RUN_NOT_VERIFIED")

    source = (file_path or (_run_dir(workspace, run_name) / "submission.csv")).expanduser()
    if not source.is_absolute():
        source = (workspace.root / source).resolve()
    else:
        source = source.resolve()
    if not source.is_file():
        raise SubmissionError(
            "SUBMISSION_ARTIFACT_MISSING: provide --file or write submission.csv in the Run output"
        )
    if source.stat().st_size == 0:
        raise SubmissionError("SUBMISSION_INVALID: submission file is empty")
    if source.suffix.lower() != ".csv":
        raise SubmissionError("SUBMISSION_INVALID: v0 tabular submissions must be CSV")

    # Allocate the durable record first so the human ID is runtime-owned.
    placeholder = create_submission_record(
        workspace.db_path,
        competition_id=workspace.competition_id,
        source_run_id=str(run["id"]),
        file_path="pending",
        file_sha256="pending",
        platform=config.competition.platform,
        message=message,
    )
    destination = _submission_dir(workspace) / f"{placeholder['name']}.csv"
    try:
        shutil.copy2(source, destination)
        digest = _sha256_file(destination)
        # The store intentionally has no general-purpose mutation API. These two
        # immutable provenance fields are completed immediately after the copy.
        import sqlite3

        with sqlite3.connect(workspace.db_path) as connection:
            connection.execute(
                "UPDATE submissions SET file_path = ?, file_sha256 = ? WHERE id = ?",
                (str(destination.resolve()), digest, str(placeholder["id"])),
            )
    except Exception:
        destination.unlink(missing_ok=True)
        import sqlite3

        with sqlite3.connect(workspace.db_path) as connection:
            connection.execute("DELETE FROM submissions WHERE id = ?", (str(placeholder["id"]),))
        raise

    record = get_submission(workspace.db_path, workspace.competition_id, str(placeholder["name"]))
    assert record is not None
    return record


def _read_csv(path: Path) -> tuple[list[str], list[list[str]]]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle)
            rows = list(reader)
    except (OSError, UnicodeError, csv.Error) as exc:
        raise SubmissionError(f"SUBMISSION_INVALID: cannot read CSV {path.name}") from exc
    if not rows:
        raise SubmissionError(f"SUBMISSION_INVALID: {path.name} is empty")
    header = [cell.strip() for cell in rows[0]]
    if not header or any(not name for name in header):
        raise SubmissionError(f"SUBMISSION_INVALID: {path.name} has invalid headers")
    body = rows[1:]
    if any(len(row) != len(header) for row in body):
        raise SubmissionError(f"SUBMISSION_INVALID: {path.name} has ragged rows")
    return header, body


def _prediction_type_for_submission(workspace: Workspace, run_name: str) -> PredictionType | None:
    path = _run_dir(workspace, run_name) / "validation.yaml"
    if not path.is_file():
        return None
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        spec = ValidationSpec.model_validate(raw)
    except Exception:
        return None
    return spec.prediction.type if spec.prediction else None


def validate_submission(
    workspace: Workspace,
    name: str,
    *,
    sample_path: Path | None = None,
) -> dict[str, object]:
    submission = get_submission(workspace.db_path, workspace.competition_id, name)
    if submission is None:
        raise SubmissionError(f"SUBMISSION_NOT_FOUND: {name}")
    if submission["status"] not in {"created", "validated"}:
        raise SubmissionError(f"SUBMISSION_INVALID_STATE: {submission['status']}")

    actual = Path(str(submission["file_path"]))
    if not actual.is_file() or _sha256_file(actual) != submission["file_sha256"]:
        raise SubmissionError("SUBMISSION_ARTIFACT_MODIFIED")

    sample = sample_path or (workspace.root / "data" / "raw" / "sample_submission.csv")
    sample = sample.expanduser()
    if not sample.is_absolute():
        sample = (workspace.root / sample).resolve()
    else:
        sample = sample.resolve()
    if not sample.is_file():
        raise SubmissionError(
            "SAMPLE_SUBMISSION_NOT_FOUND: expected data/raw/sample_submission.csv or --sample"
        )

    sample_header, sample_rows = _read_csv(sample)
    actual_header, actual_rows = _read_csv(actual)
    if actual_header != sample_header:
        raise SubmissionError(
            "SUBMISSION_SCHEMA_MISMATCH: headers differ from sample_submission.csv"
        )
    if len(actual_rows) != len(sample_rows):
        raise SubmissionError(
            f"SUBMISSION_ROW_COUNT_MISMATCH: expected {len(sample_rows)}, got {len(actual_rows)}"
        )
    if len(actual_header) < 2:
        raise SubmissionError("SUBMISSION_INVALID: expected at least ID and prediction columns")

    sample_ids = [row[0] for row in sample_rows]
    actual_ids = [row[0] for row in actual_rows]
    if actual_ids != sample_ids:
        raise SubmissionError("SUBMISSION_ID_MISMATCH: first-column IDs/order differ from sample")
    if len(set(actual_ids)) != len(actual_ids):
        raise SubmissionError("SUBMISSION_INVALID: duplicate first-column IDs")

    prediction_type = _prediction_type_for_submission(
        workspace, str(submission["source_run_name"])
    )
    for row_number, row in enumerate(actual_rows, start=2):
        for column_number, value in enumerate(row[1:], start=2):
            stripped = value.strip()
            if not stripped:
                raise SubmissionError(
                    f"SUBMISSION_INVALID: blank prediction at row {row_number}, column {column_number}"
                )
            if prediction_type in {PredictionType.PROBABILITY, PredictionType.VALUE}:
                try:
                    number = float(stripped)
                except ValueError as exc:
                    raise SubmissionError(
                        f"SUBMISSION_INVALID: non-numeric prediction at row {row_number}, column {column_number}"
                    ) from exc
                if not math.isfinite(number):
                    raise SubmissionError(
                        f"SUBMISSION_INVALID: non-finite prediction at row {row_number}, column {column_number}"
                    )
                if prediction_type == PredictionType.PROBABILITY and not 0.0 <= number <= 1.0:
                    raise SubmissionError(
                        f"SUBMISSION_INVALID: probability outside [0, 1] at row {row_number}, column {column_number}"
                    )

    return mark_validated(workspace.db_path, workspace.competition_id, name)


def budget_status(
    workspace: Workspace,
    *,
    provider: SubmissionProvider | None = None,
) -> dict[str, object]:
    config = load_arena_config(workspace)
    usage = submission_budget_usage(workspace.db_path, workspace.competition_id)
    daily_budget = config.submission.daily_budget
    total_budget = config.submission.total_budget
    provider_limits = None
    if provider is not None:
        provider_limits = provider.submission_limits(config.competition.slug)
    return {
        "daily_budget": daily_budget,
        "daily_used": usage["daily"],
        "daily_remaining": None if daily_budget == 0 else max(daily_budget - usage["daily"], 0),
        "total_budget": total_budget,
        "total_used": usage["total"],
        "total_remaining": None if total_budget == 0 else max(total_budget - usage["total"], 0),
        "provider_limits": provider_limits,
    }


def _enforce_local_budget(workspace: Workspace) -> None:
    config = load_arena_config(workspace)
    usage = submission_budget_usage(workspace.db_path, workspace.competition_id)
    if config.submission.daily_budget and usage["daily"] >= config.submission.daily_budget:
        raise SubmissionError("SUBMISSION_BUDGET_EXCEEDED: daily budget exhausted")
    if config.submission.total_budget and usage["total"] >= config.submission.total_budget:
        raise SubmissionError("SUBMISSION_BUDGET_EXCEEDED: total budget exhausted")


def send_submission(
    workspace: Workspace,
    name: str,
    *,
    message: str | None = None,
    provider: SubmissionProvider | None = None,
) -> dict[str, object]:
    config = load_arena_config(workspace)
    submission = get_submission(workspace.db_path, workspace.competition_id, name)
    if submission is None:
        raise SubmissionError(f"SUBMISSION_NOT_FOUND: {name}")
    if submission["status"] != "validated":
        raise SubmissionError("SUBMISSION_NOT_VALIDATED")
    _enforce_local_budget(workspace)

    file_path = Path(str(submission["file_path"]))
    if not file_path.is_file() or _sha256_file(file_path) != submission["file_sha256"]:
        raise SubmissionError("SUBMISSION_ARTIFACT_MODIFIED")
    resolved_message = (message or submission.get("message") or f"ArenaPilot {name}").strip()
    if not resolved_message:
        raise SubmissionError("SUBMISSION_MESSAGE_REQUIRED")

    adapter = provider or KaggleSubmissionProvider()
    submission_ref = adapter.submit_file(config.competition.slug, file_path, resolved_message)
    return mark_submitted(
        workspace.db_path,
        competition_id=workspace.competition_id,
        name=name,
        platform_submission_id=submission_ref,
        message=resolved_message,
    )


def _as_optional_float(value: object) -> float | None:
    if value in {None, "", "null", "None"}:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def sync_submission(
    workspace: Workspace,
    name: str,
    *,
    provider: SubmissionProvider | None = None,
) -> dict[str, object]:
    config = load_arena_config(workspace)
    submission = get_submission(workspace.db_path, workspace.competition_id, name)
    if submission is None:
        raise SubmissionError(f"SUBMISSION_NOT_FOUND: {name}")
    ref = submission.get("platform_submission_id")
    if not ref:
        return submission

    adapter = provider or KaggleSubmissionProvider()
    state = adapter.submission_status(config.competition.slug, str(ref))
    raw_status = str(state.get("status") or state.get("state") or "pending").lower()
    public_score = _as_optional_float(
        state.get("publicScore") if "publicScore" in state else state.get("public_score")
    )
    private_score = _as_optional_float(
        state.get("privateScore") if "privateScore" in state else state.get("private_score")
    )

    if raw_status in {"error", "failed", "failure"}:
        status = "failed"
        failure = str(state.get("error") or state.get("message") or raw_status)
    elif raw_status in {"complete", "completed", "finished", "scored"} or public_score is not None:
        status = "scored"
        failure = None
    else:
        status = "pending"
        failure = None
    return update_submission_score(
        workspace.db_path,
        competition_id=workspace.competition_id,
        name=name,
        status=status,
        public_score=public_score,
        private_score=private_score,
        failure_message=failure,
    )


def list_submission_summaries(
    workspace: Workspace,
    *,
    sync: bool = False,
    provider: SubmissionProvider | None = None,
) -> list[dict[str, object]]:
    rows = list_submissions(workspace.db_path, workspace.competition_id)
    if sync:
        refreshed: list[dict[str, object]] = []
        for row in rows:
            if row.get("platform_submission_id"):
                refreshed.append(sync_submission(workspace, str(row["name"]), provider=provider))
            else:
                refreshed.append(row)
        rows = refreshed
    return [
        {
            "id": row["name"],
            "experiment": row["experiment_name"],
            "run": row["source_run_name"],
            "status": row["status"],
            "platform_submission_id": row["platform_submission_id"],
            "public_score": row["public_score"],
            "private_score": row["private_score"],
            "message": row["message"],
            "file_sha256": row["file_sha256"],
        }
        for row in rows
    ]
