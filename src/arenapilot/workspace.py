from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import yaml

from .db import initialize_database
from .models import ArenaConfig, ValidationSpec


class WorkspaceError(RuntimeError):
    pass


class WorkspaceNotFoundError(WorkspaceError):
    pass


class WorkspaceExistsError(WorkspaceError):
    pass


@dataclass(frozen=True)
class Workspace:
    root: Path
    competition_id: str
    workspace_id: str

    @property
    def state_dir(self) -> Path:
        return self.root / ".arena"

    @property
    def db_path(self) -> Path:
        return self.state_dir / "arena.db"

    @property
    def config_path(self) -> Path:
        return self.root / "arena.yaml"

    def validation_path(self, name: str) -> Path:
        return self.root / "configs" / "validation" / f"{name}.yaml"


SCAFFOLD_DIRS = (
    "configs/validation",
    "configs/models",
    "configs/pipelines",
    "experiments",
    "src",
    "data/raw",
    "data/processed",
    "outputs/runs",
    "submissions",
    "reports",
    "notebooks",
    ".arena",
)


def parse_competition_ref(value: str) -> tuple[str, str]:
    try:
        platform, slug = value.split(":", 1)
    except ValueError as exc:
        raise WorkspaceError("competition must use <platform>:<slug>, e.g. kaggle:titanic") from exc
    if platform != "kaggle":
        raise WorkspaceError(f"unsupported competition platform: {platform}")
    if not slug.strip():
        raise WorkspaceError("competition slug cannot be empty")
    return platform, slug.strip()


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex}"


def _yaml_text(model: ArenaConfig | ValidationSpec) -> str:
    payload = model.model_dump(mode="json", exclude_none=False)
    return yaml.safe_dump(payload, sort_keys=False, allow_unicode=True)


def _atomic_write_text(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def save_arena_config(workspace: Workspace, config: ArenaConfig) -> None:
    _atomic_write_text(workspace.config_path, _yaml_text(config))


def save_validation_spec(workspace: Workspace, spec: ValidationSpec) -> None:
    _atomic_write_text(workspace.validation_path(spec.id), _yaml_text(spec))


def load_arena_config(workspace: Workspace) -> ArenaConfig:
    raw = yaml.safe_load(workspace.config_path.read_text(encoding="utf-8"))
    return ArenaConfig.model_validate(raw)


def load_validation_spec(workspace: Workspace, name: str) -> ValidationSpec:
    path = workspace.validation_path(name)
    if not path.is_file():
        raise WorkspaceError(f"validation not found: {name}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return ValidationSpec.model_validate(raw)


def create_workspace(
    competition_ref: str,
    destination: Path | None = None,
    *,
    title: str | None = None,
) -> Workspace:
    platform, slug = parse_competition_ref(competition_ref)
    destination = (destination or Path(slug)).expanduser().resolve()
    if destination.exists():
        raise WorkspaceExistsError(f"destination already exists: {destination}")

    parent = destination.parent
    parent.mkdir(parents=True, exist_ok=True)
    staging = parent / f".{destination.name}.arenapilot-{uuid4().hex}.tmp"
    competition_id = _new_id("cmp")
    workspace_id = _new_id("ws")

    try:
        staging.mkdir()
        for relative in SCAFFOLD_DIRS:
            (staging / relative).mkdir(parents=True, exist_ok=True)

        config = ArenaConfig.model_validate(
            {
                "schema_version": 1,
                "competition": {
                    "platform": platform,
                    "slug": slug,
                    "title": title,
                    "status": "draft",
                },
                "task": None,
                "metric": None,
                "validation": {"active": None},
                "tracking": {
                    "backend": "mlflow",
                    "experiment_name": slug,
                },
            }
        )
        (staging / "arena.yaml").write_text(_yaml_text(config), encoding="utf-8")

        validation = ValidationSpec.model_validate(
            {
                "schema_version": 1,
                "id": "val-v1",
                "parent": None,
                "status": "draft",
                "reason": "initial_validation",
                "split": {
                    "type": "kfold",
                    "n_splits": 5,
                    "shuffle": True,
                    "random_state": 42,
                    "group_column": None,
                    "time_column": None,
                },
                "metric": None,
                "prediction": None,
                "oof": {"required": True, "require_exactly_once": True},
                "metadata": {
                    "notes": "Draft validation. Configure after competition intake before activation."
                },
            }
        )
        (staging / "configs" / "validation" / "val-v1.yaml").write_text(
            _yaml_text(validation),
            encoding="utf-8",
        )

        marker = {
            "schema_version": 1,
            "competition_id": competition_id,
            "workspace_id": workspace_id,
        }
        (staging / ".arena" / "workspace.json").write_text(
            json.dumps(marker, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        initialize_database(staging / ".arena" / "arena.db")

        (staging / "src" / "__init__.py").write_text("", encoding="utf-8")
        (staging / "README.md").write_text(
            f"# {title or slug}\n\nArenaPilot workspace for `{platform}:{slug}`.\n",
            encoding="utf-8",
        )
        (staging / ".gitignore").write_text(
            "data/raw/*\ndata/processed/*\noutputs/runs/*\nsubmissions/*\n!.gitkeep\n",
            encoding="utf-8",
        )
        for relative in (
            "data/raw/.gitkeep",
            "data/processed/.gitkeep",
            "outputs/runs/.gitkeep",
            "submissions/.gitkeep",
            "experiments/.gitkeep",
            "reports/.gitkeep",
            "notebooks/.gitkeep",
        ):
            (staging / relative).write_text("", encoding="utf-8")

        staging.rename(destination)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        raise

    return Workspace(
        root=destination,
        competition_id=competition_id,
        workspace_id=workspace_id,
    )


def discover_workspace(start: Path | None = None) -> Workspace:
    current = (start or Path.cwd()).resolve()
    if current.is_file():
        current = current.parent

    for candidate in (current, *current.parents):
        marker = candidate / ".arena" / "workspace.json"
        if marker.is_file():
            data = json.loads(marker.read_text(encoding="utf-8"))
            if data.get("schema_version") != 1:
                raise ValueError(f"unsupported workspace schema in {marker}")
            return Workspace(
                root=candidate,
                competition_id=data["competition_id"],
                workspace_id=data["workspace_id"],
            )

    raise WorkspaceNotFoundError(f"no ArenaPilot workspace found from {current}")
