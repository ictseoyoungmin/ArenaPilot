from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from uuid import uuid4

from .db import get_experiment
from .experiments import ExperimentError, freeze_experiment
from .remote_store import (
    create_remote_job,
    get_remote_job_for_run,
    update_remote_job,
)
from .runstore import create_run_record, get_run, transition_run
from .workspace import Workspace, WorkspaceError, load_arena_config, load_experiment_spec


class KaggleBackendError(WorkspaceError):
    pass


def _run_dir(workspace: Workspace, run_name: str) -> Path:
    return workspace.root / "outputs" / "runs" / run_name


def _bundle_dir(workspace: Workspace, run_name: str) -> Path:
    return workspace.state_dir / "bundles" / run_name


def _legacy_kaggle_username() -> str | None:
    config_dir = Path(os.environ.get("KAGGLE_CONFIG_DIR", "~/.kaggle")).expanduser()
    path = config_dir / "kaggle.json"
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    value = payload.get("username") if isinstance(payload, dict) else None
    return str(value).strip() if value else None


def resolve_kaggle_owner(explicit: str | None = None) -> str:
    owner = (
        explicit
        or os.environ.get("ARENA_KAGGLE_OWNER")
        or os.environ.get("KAGGLE_USERNAME")
        or _legacy_kaggle_username()
    )
    if not owner or not owner.strip():
        raise KaggleBackendError(
            "KAGGLE_OWNER_REQUIRED: set ARENA_KAGGLE_OWNER or KAGGLE_USERNAME"
        )
    return owner.strip()


def _kaggle_executable() -> str:
    executable = shutil.which("kaggle")
    if executable is None:
        raise KaggleBackendError("KAGGLE_CLI_NOT_FOUND: install the official kaggle CLI")
    return executable


def _run_kaggle_cli(args: list[str]) -> subprocess.CompletedProcess[str]:
    process = subprocess.run(
        [_kaggle_executable(), *args],
        capture_output=True,
        text=True,
        check=False,
    )
    if process.returncode != 0:
        detail = (process.stderr or process.stdout or "unknown Kaggle CLI error").strip()
        raise KaggleBackendError(f"KAGGLE_CLI_FAILED: {detail}")
    return process


def _safe_kernel_slug(competition_slug: str, run_name: str) -> str:
    raw = f"arenapilot-{competition_slug}-{run_name}".lower()
    slug = re.sub(r"[^a-z0-9-]+", "-", raw).strip("-")
    slug = re.sub(r"-+", "-", slug)
    if len(slug) <= 48:
        return slug
    digest = hashlib.sha256(slug.encode("utf-8")).hexdigest()[:8]
    return f"{slug[:39].rstrip('-')}-{digest}"


def _encoded_workspace_source(workspace: Workspace) -> dict[str, str]:
    source_root = workspace.root / "src"
    if not source_root.is_dir():
        raise KaggleBackendError("workspace src directory is missing")

    encoded: dict[str, str] = {}
    total_bytes = 0
    for path in sorted(source_root.rglob("*")):
        if not path.is_file():
            continue
        if "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
            continue
        payload = path.read_bytes()
        total_bytes += len(payload)
        if total_bytes > 20 * 1024 * 1024:
            raise KaggleBackendError("Kaggle source bundle exceeds 20 MiB")
        relative = str(path.relative_to(workspace.root)).replace(os.sep, "/")
        encoded[relative] = base64.b64encode(payload).decode("ascii")
    return encoded


def _runner_script(
    *,
    source_files: dict[str, str],
    experiment_yaml: str,
    validation_yaml: str,
    competition_slug: str,
    run_name: str,
) -> str:
    files_json = json.dumps(source_files, sort_keys=True)
    return f'''from __future__ import annotations

import base64
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

SOURCE_FILES = json.loads({files_json!r})
EXPERIMENT_YAML = {experiment_yaml!r}
VALIDATION_YAML = {validation_yaml!r}
COMPETITION_SLUG = {competition_slug!r}
RUN_NAME = {run_name!r}


def main() -> int:
    output = Path("/kaggle/working")
    output.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="arenapilot-"))
    try:
        for relative, encoded in SOURCE_FILES.items():
            target = work / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(base64.b64decode(encoded.encode("ascii")))

        spec_path = output / "spec.yaml"
        validation_path = output / "validation.yaml"
        spec_path.write_text(EXPERIMENT_YAML, encoding="utf-8")
        validation_path.write_text(VALIDATION_YAML, encoding="utf-8")
        (output / "environment.json").write_text(
            json.dumps(
                {{
                    "schema_version": 1,
                    "python": platform.python_version(),
                    "platform": platform.platform(),
                    "backend": "kaggle",
                    "executable": sys.executable,
                }},
                indent=2,
                sort_keys=True,
            ) + "\\n",
            encoding="utf-8",
        )

        env = os.environ.copy()
        env.update(
            {{
                "ARENA_RUN_ID": RUN_NAME,
                "ARENA_EXPERIMENT_SPEC": str(spec_path),
                "ARENA_VALIDATION_SPEC": str(validation_path),
                "ARENA_OUTPUT_DIR": str(output),
                "ARENA_DATA_DIR": str(Path("/kaggle/input") / COMPETITION_SLUG),
            }}
        )
        process = subprocess.run(
            [sys.executable, "-m", "src.train"],
            cwd=work,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        log_text = process.stdout
        if process.stderr:
            if log_text and not log_text.endswith("\\n"):
                log_text += "\\n"
            log_text += process.stderr
        (output / "logs.txt").write_text(log_text, encoding="utf-8")
        return process.returncode
    except Exception:
        (output / "logs.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
'''


def prepare_kaggle_bundle(
    workspace: Workspace,
    *,
    run_name: str,
    experiment_name: str,
    owner: str,
) -> tuple[Path, str]:
    config = load_arena_config(workspace)
    spec = load_experiment_spec(workspace, experiment_name)
    bundle = _bundle_dir(workspace, run_name)
    if bundle.exists():
        raise KaggleBackendError(f"remote bundle already exists: {bundle}")
    bundle.mkdir(parents=True, exist_ok=False)

    experiment_path = workspace.experiment_path(experiment_name)
    validation_path = workspace.validation_path(spec.validation)
    if not validation_path.is_file():
        raise KaggleBackendError(f"validation snapshot source missing: {spec.validation}")

    experiment_yaml = experiment_path.read_text(encoding="utf-8")
    validation_yaml = validation_path.read_text(encoding="utf-8")
    source_files = _encoded_workspace_source(workspace)
    runner = _runner_script(
        source_files=source_files,
        experiment_yaml=experiment_yaml,
        validation_yaml=validation_yaml,
        competition_slug=config.competition.slug,
        run_name=run_name,
    )
    (bundle / "arena_kernel.py").write_text(runner, encoding="utf-8")
    (bundle / "experiment.yaml").write_text(experiment_yaml, encoding="utf-8")
    (bundle / "validation.yaml").write_text(validation_yaml, encoding="utf-8")

    kernel_slug = _safe_kernel_slug(config.competition.slug, run_name)
    kernel_ref = f"{owner}/{kernel_slug}"
    accelerator = config.compute.kaggle.accelerator
    metadata = {
        "id": kernel_ref,
        "title": kernel_slug,
        "code_file": "arena_kernel.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": bool(accelerator),
        "enable_tpu": False,
        "enable_internet": config.compute.kaggle.internet,
        "dataset_sources": [],
        "competition_sources": [config.competition.slug],
        "kernel_sources": [],
        "model_sources": [],
    }
    (bundle / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return bundle, kernel_ref


def _push_bundle(workspace: Workspace, bundle: Path) -> None:
    config = load_arena_config(workspace)
    args = ["kernels", "push", "-p", str(bundle)]
    accelerator = config.compute.kaggle.accelerator
    if accelerator and accelerator.lower() != "gpu":
        args.extend(["--accelerator", accelerator])
    _run_kaggle_cli(args)


def _parse_kernel_status(output: str) -> str:
    value = output.lower()
    if any(token in value for token in ("error", "failed", "cancelled", "canceled")):
        return "failed"
    if re.search(r"\bcomplete(?:d)?\b", value):
        return "completed"
    if "running" in value:
        return "running"
    if "queued" in value or "pending" in value:
        return "queued"
    return "unknown"


def _remote_job_for_run(workspace: Workspace, run_name: str) -> tuple[dict[str, object], dict[str, object]]:
    run = get_run(workspace.db_path, workspace.competition_id, run_name)
    if run is None:
        raise KaggleBackendError(f"run not found: {run_name}")
    if run["backend"] != "kaggle":
        raise KaggleBackendError(f"run is not a Kaggle run: {run_name}")
    job = get_remote_job_for_run(workspace.db_path, str(run["id"]))
    if job is None:
        raise KaggleBackendError(f"remote job not found for run: {run_name}")
    return run, job


def sync_kaggle_status(workspace: Workspace, run_name: str) -> dict[str, object]:
    run, job = _remote_job_for_run(workspace, run_name)
    kernel_ref = str(job["provider_job_id"])
    process = _run_kaggle_cli(["kernels", "status", kernel_ref])
    raw_status = (process.stdout or process.stderr).strip()
    state = _parse_kernel_status(raw_status)
    if state == "unknown":
        raise KaggleBackendError(f"REMOTE_JOB_AMBIGUOUS: {raw_status}")

    job = update_remote_job(workspace.db_path, str(job["id"]), state=state)
    current = str(run["status"])
    if state == "running" and current == "queued":
        run = transition_run(
            workspace.db_path,
            competition_id=workspace.competition_id,
            name=run_name,
            from_statuses={"queued"},
            to_status="running",
        )
    elif state == "completed" and current in {"queued", "running"}:
        run = transition_run(
            workspace.db_path,
            competition_id=workspace.competition_id,
            name=run_name,
            from_statuses={current},
            to_status="completed",
            exit_code=0,
        )
    elif state == "failed" and current in {"queued", "running"}:
        run = transition_run(
            workspace.db_path,
            competition_id=workspace.competition_id,
            name=run_name,
            from_statuses={current},
            to_status="failed",
            exit_code=1,
        )

    return {
        "run": run,
        "remote_job": job,
        "provider_status": state,
        "raw_status": raw_status,
    }


def kaggle_remote_logs(workspace: Workspace, run_name: str) -> str:
    _, job = _remote_job_for_run(workspace, run_name)
    process = _run_kaggle_cli(["kernels", "logs", str(job["provider_job_id"])])
    text = process.stdout
    if process.stderr:
        if text and not text.endswith("\n"):
            text += "\n"
        text += process.stderr
    return text


def _download_remote_output(workspace: Workspace, run_name: str, kernel_ref: str) -> Path:
    recovery_root = workspace.state_dir / "recovery"
    recovery_root.mkdir(parents=True, exist_ok=True)
    staging = recovery_root / f".{run_name}.{uuid4().hex}.tmp"
    staging.mkdir(parents=True, exist_ok=False)
    try:
        _run_kaggle_cli(
            ["kernels", "output", kernel_ref, "-p", str(staging), "-o", "-q"]
        )
        result_files = list(staging.rglob("result.json"))
        source = result_files[0].parent if len(result_files) == 1 else staging

        target = _run_dir(workspace, run_name)
        assembled = target.parent / f".{run_name}.{uuid4().hex}.tmp"
        if assembled.exists():
            shutil.rmtree(assembled)
        shutil.copytree(source, assembled)
        if target.exists():
            shutil.rmtree(target)
        assembled.rename(target)
        return target
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def recover_kaggle_run(workspace: Workspace, run_name: str) -> dict[str, object]:
    status = sync_kaggle_status(workspace, run_name)
    run = status["run"]
    job = status["remote_job"]
    provider_status = str(status["provider_status"])

    if provider_status == "failed":
        try:
            logs = kaggle_remote_logs(workspace, run_name)
            run_dir = _run_dir(workspace, run_name)
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "logs.txt").write_text(logs, encoding="utf-8")
        except Exception:
            pass
        update_remote_job(
            workspace.db_path,
            str(job["id"]),
            recovery_state="failed",
        )
        raise KaggleBackendError(f"REMOTE_JOB_FAILED: {job['provider_job_id']}")

    if provider_status != "completed":
        raise KaggleBackendError(
            f"REMOTE_JOB_NOT_READY: {job['provider_job_id']} is {provider_status}"
        )

    if str(run["status"]) == "verified":
        update_remote_job(
            workspace.db_path,
            str(job["id"]),
            recovery_state="verified",
        )
        return run

    try:
        _download_remote_output(workspace, run_name, str(job["provider_job_id"]))
        update_remote_job(
            workspace.db_path,
            str(job["id"]),
            recovery_state="pulled",
        )
        from .runs import verify_run

        verified = verify_run(workspace, run_name)
        update_remote_job(
            workspace.db_path,
            str(job["id"]),
            recovery_state="verified",
        )
        return verified
    except Exception:
        try:
            update_remote_job(
                workspace.db_path,
                str(job["id"]),
                recovery_state="failed",
            )
        except Exception:
            pass
        raise


def wait_for_kaggle_run(
    workspace: Workspace,
    run_name: str,
    *,
    poll_seconds: float = 15.0,
    timeout_seconds: float = 43200.0,
) -> dict[str, object]:
    deadline = time.monotonic() + timeout_seconds
    while True:
        status = sync_kaggle_status(workspace, run_name)
        provider_status = str(status["provider_status"])
        if provider_status == "completed":
            return recover_kaggle_run(workspace, run_name)
        if provider_status == "failed":
            return recover_kaggle_run(workspace, run_name)
        if time.monotonic() >= deadline:
            raise KaggleBackendError(f"REMOTE_JOB_TIMEOUT: {run_name}")
        time.sleep(max(0.1, poll_seconds))


def run_kaggle_experiment(
    workspace: Workspace,
    experiment_name: str,
    *,
    owner: str | None = None,
    wait: bool = False,
    poll_seconds: float = 15.0,
    timeout_seconds: float = 43200.0,
) -> dict[str, object]:
    experiment_row = get_experiment(
        workspace.db_path,
        workspace.competition_id,
        experiment_name,
    )
    if experiment_row is None:
        raise KaggleBackendError(f"experiment not found: {experiment_name}")
    if experiment_row["status"] != "frozen":
        raise KaggleBackendError("EXPERIMENT_NOT_FROZEN")

    try:
        freeze_experiment(workspace, experiment_name)
    except ExperimentError as exc:
        raise KaggleBackendError(str(exc)) from exc

    config = load_arena_config(workspace)
    if not config.compute.kaggle.enabled:
        raise KaggleBackendError("Kaggle backend is disabled in arena.yaml")
    resolved_owner = resolve_kaggle_owner(owner)

    record = create_run_record(
        workspace.db_path,
        competition_id=workspace.competition_id,
        experiment_id=str(experiment_row["id"]),
        backend="kaggle",
        spec_hash=str(experiment_row["config_hash"]),
    )
    run_name = str(record["name"])

    try:
        bundle, kernel_ref = prepare_kaggle_bundle(
            workspace,
            run_name=run_name,
            experiment_name=experiment_name,
            owner=resolved_owner,
        )
        job = create_remote_job(
            workspace.db_path,
            run_id=str(record["id"]),
            provider="kaggle",
            provider_job_id=kernel_ref,
            bundle_path=str(bundle),
        )
        _push_bundle(workspace, bundle)
        update_remote_job(workspace.db_path, str(job["id"]), state="submitted")
        record = transition_run(
            workspace.db_path,
            competition_id=workspace.competition_id,
            name=run_name,
            from_statuses={"created"},
            to_status="queued",
        )
    except Exception as exc:
        try:
            transition_run(
                workspace.db_path,
                competition_id=workspace.competition_id,
                name=run_name,
                from_statuses={"created"},
                to_status="failed",
                exit_code=-1,
            )
        except Exception:
            pass
        raise KaggleBackendError(str(exc)) from exc

    if wait:
        return wait_for_kaggle_run(
            workspace,
            run_name,
            poll_seconds=poll_seconds,
            timeout_seconds=timeout_seconds,
        )
    return record
