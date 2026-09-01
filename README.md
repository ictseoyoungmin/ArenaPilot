# ArenaPilot

ArenaPilot is a local-first agent runtime for reproducible machine-learning competition workflows.

> Agents may forget. ArenaPilot should not.

## Core contract

- **Experiment != Run**: an experiment is intent; every execution creates a new run.
- **Validation is versioned**: scores are only directly comparable inside a compatible validation domain.
- **SQLite != MLflow**: ArenaPilot owns operational state; MLflow owns experiment telemetry and artifacts.
- **Memory is evidence-based**: cross-competition knowledge must retain supporting and contradicting evidence.
- **CLI owns side effects**: agents edit declarative specs and use `arena` for mutations.

## Bootstrap a competition workspace

```bash
arena init kaggle:titanic
cd titanic
arena status
arena doctor
```

`arena init` intentionally creates a **draft** workspace. At init time ArenaPilot may not yet know the target, metric, prediction semantics, or leakage-safe validation strategy. It therefore creates `arena.yaml` and `configs/validation/val-v1.yaml` without inventing those values. Competition intake will promote the configuration to `ready` in a later slice.

The workspace contract is:

```text
competition/
├── arena.yaml
├── configs/
│   ├── validation/val-v1.yaml
│   ├── models/
│   └── pipelines/
├── experiments/
├── src/
├── data/{raw,processed}/
├── outputs/runs/
├── submissions/
├── reports/
├── notebooks/
└── .arena/
    ├── workspace.json
    └── arena.db
```

Creation is staged in a sibling temporary directory and renamed into place only after all required files and the SQLite database are initialized, so a failed init does not leave a partially-created workspace at the requested destination.

## Development

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest -q
arena version
```

Kaggle data intake, validation activation, experiment execution, and MLflow ingestion remain subsequent slices.
