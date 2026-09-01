# ArenaPilot

ArenaPilot is a local-first agent runtime for reproducible machine-learning competition workflows.

> Agents may forget. ArenaPilot should not.

## Core contract

- **Experiment != Run**: an experiment is intent; every execution creates a new run.
- **Validation is versioned**: scores are only directly comparable inside a compatible validation domain.
- **SQLite != MLflow**: ArenaPilot owns operational state; MLflow owns experiment telemetry and artifacts.
- **Memory is evidence-based**: cross-competition knowledge must retain supporting and contradicting evidence.
- **CLI owns side effects**: agents edit declarative specs and use `arena` for mutations.

## Bootstrap and configure a competition workspace

```bash
arena init kaggle:titanic
cd titanic

arena intake set \
  --task binary_classification \
  --target Survived \
  --metric roc_auc \
  --direction maximize

arena validation configure val-v1 \
  --split stratified_kfold \
  --prediction probability

arena validation activate val-v1
arena status
arena doctor
```

`arena init` intentionally creates a **draft** workspace. At init time ArenaPilot may not yet know the target, metric, prediction semantics, or leakage-safe validation strategy. `arena intake set` records the competition task contract, and `arena validation configure` fills the draft validation using that primary metric. Only a complete, compatible validation can be activated; activation promotes the competition to `ready` and records validation/spec hashes in SQLite.

Once a validation is active, the competition intake contract is immutable in v0. Changing target or metric later must be handled as an explicit migration rather than silently rewriting the evaluation history.

## Create and freeze experiments

Experiments can only be created after the competition is `ready`. Each experiment is automatically bound to the active validation and receives the next stable human-readable ID.

```bash
arena exp new \
  --title baseline \
  --hypothesis "A CatBoost baseline establishes the comparison floor." \
  --model-family catboost

# edit experiments/exp001.yaml while it is still draft
arena exp freeze exp001

arena exp show exp001
arena exp list
```

A frozen experiment is immutable. ArenaPilot stores its config hash in SQLite and an immutable snapshot under `outputs/specs/<experiment>/<hash>.yaml`. If the editable experiment YAML is changed after freeze, integrity checks report the mismatch and another freeze fails with `FROZEN_SPEC_MODIFIED`.

Derived experiments require a frozen parent:

```bash
arena exp new \
  --from exp001 \
  --title frequency-encoding \
  --hypothesis "Frequency encoding improves high-cardinality categoricals." \
  --model-family catboost

arena exp lineage exp002
```

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
├── outputs/
│   ├── runs/
│   └── specs/
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

Run execution, artifact verification, Kaggle compute, and MLflow ingestion remain subsequent slices.
