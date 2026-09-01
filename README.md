# ArenaPilot

ArenaPilot is a local-first agent runtime for reproducible machine-learning competition workflows.

> Agents may forget. ArenaPilot should not.

## Core contract

ArenaPilot separates four concerns from the start:

- **Experiment vs Run** — an experiment captures intent; every execution becomes a distinct run.
- **Versioned validation** — experiments freeze the evaluation context they were designed against.
- **Arena DB vs MLflow** — SQLite owns operational state; MLflow will own metrics and experiment telemetry.
- **Cross-competition memory** — evidence can later be promoted into reusable, contextual competition knowledge.

The CLI is the supported mutation boundary. Declarative YAML is agent-editable; runtime state is not.

## Development

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest -q
arena version
```

The first implementation slice establishes the v0 schemas, workspace discovery, SQLite bootstrap, and CLI foundation. Kaggle execution and MLflow ingestion are intentionally subsequent slices.
