from __future__ import annotations

from .kaggle_backend import run_kaggle_experiment
from .runs import RunError, run_local_experiment
from .workspace import Workspace, load_experiment_spec


def run_experiment(
    workspace: Workspace,
    experiment_name: str,
    *,
    backend: str | None = None,
    kaggle_owner: str | None = None,
    wait: bool = False,
    poll_seconds: float = 15.0,
    timeout_seconds: float = 43200.0,
) -> dict[str, object]:
    spec = load_experiment_spec(workspace, experiment_name)
    selected_backend = backend or spec.runtime.backend
    if selected_backend == "local":
        return run_local_experiment(workspace, experiment_name, backend="local")
    if selected_backend == "kaggle":
        return run_kaggle_experiment(
            workspace,
            experiment_name,
            owner=kaggle_owner,
            wait=wait,
            poll_seconds=poll_seconds,
            timeout_seconds=timeout_seconds,
        )
    raise RunError(f"unsupported execution backend: {selected_backend}")
