from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


class WorkspaceNotFoundError(RuntimeError):
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
