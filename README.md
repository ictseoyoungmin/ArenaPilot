# ArenaPilot

ArenaPilot is a local-first agent runtime for reproducible machine-learning competition workflows.

> Agents may forget. ArenaPilot should not.

## Core contract

- **Experiment != Run**: an experiment is intent; every execution creates a new run.
- **Validation is versioned**: scores are only directly comparable inside a compatible validation domain.
- **SQLite != MLflow**: ArenaPilot owns operational state; MLflow owns experiment telemetry and copied artifacts.
- **Memory is evidence-based**: cross-competition knowledge must retain supporting and contradicting evidence.
- **CLI owns mutations**: agents use the `arena` CLI for runtime-managed state/spec changes; SQLite, MLflow, Kaggle provider writes, and runtime-managed YAML are not agent mutation surfaces.

## Agent Skill contract

ArenaPilot ships an agent skill under `skills/arenapilot/`. The skill owns scientific judgment while the runtime owns state, verification, tracking, provider side effects, and memory persistence.

Before material agent work in an existing workspace, the Skill can verify the runtime handshake with:

```bash
arena contract --json
arena status --json
arena doctor --json
```

The current Skill requires Agent Contract v1. `skills/arenapilot/contract.yaml` is machine-readable and CI checks that its documented public command paths remain present.

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

arena exp configure exp001 \
  --model-params-json '{"depth":8,"learning_rate":0.04}' \
  --pipeline-json '{"features":{"frequency_encoding":true}}' \
  --seed 42

arena exp show exp001
arena exp freeze exp001
arena exp list
```

`arena exp configure` is the supported agent authoring surface for draft model params, pipeline, seed, backend/resources, and tags. Validation binding and parent lineage remain fixed from creation. Once the Experiment is frozen, configure fails with `EXPERIMENT_CONFIG_IMMUTABLE`; create a new or derived Experiment instead of changing the frozen declaration.

A frozen experiment is immutable. ArenaPilot stores its config hash in SQLite and an immutable snapshot under `outputs/specs/<experiment>/<hash>.yaml`. If the persisted declaration differs after freeze, integrity checks report the mismatch and another freeze fails with `FROZEN_SPEC_MODIFIED`.

Derived experiments require a frozen parent:

```bash
arena exp new \
  --from exp001 \
  --title frequency-encoding \
  --hypothesis "Frequency encoding improves high-cardinality categoricals." \
  --model-family catboost

arena exp lineage exp002
```

## Execute local runs

A frozen experiment can be executed locally with:

```bash
arena exp run exp001
```

Every execution creates a distinct Run (`run001`, `run002`, ...). ArenaPilot launches the workspace entrypoint as:

```bash
python -m src.train
```

The training process receives these environment variables:

```text
ARENA_RUN_ID
ARENA_EXPERIMENT_SPEC
ARENA_VALIDATION_SPEC
ARENA_OUTPUT_DIR
ARENA_DATA_DIR
```

The process writes its output into `ARENA_OUTPUT_DIR`. ArenaPilot itself snapshots `spec.yaml`, `validation.yaml`, `environment.json`, and captured `logs.txt` into the same run directory.

For a normal cross-validation run, verification requires:

```text
outputs/runs/run001/
├── spec.yaml
├── validation.yaml
├── environment.json
├── result.json
├── metrics.json
├── fold_metrics.json
├── predictions.parquet
├── oof.parquet
├── submission.csv       # optional provider-ready artifact
├── logs.txt
└── manifest.json
```

`result.json` must declare `status: success` and a finite primary metric matching the validation contract. `metrics.json` must contain that metric, and `fold_metrics.json` must contain the configured number of folds. Prediction/OOF artifacts must exist and be non-empty.

The lifecycle distinguishes execution, artifact verification, and tracking:

```text
CREATED
  ↓
RUNNING
  ├────────→ FAILED       process failed
  ↓
COMPLETED
  ↓
VERIFYING
  ├────────→ INVALID      artifact contract failed
  ↓
MLflow ingest
  ├────────→ COMPLETED    tracking failed; retryable
  ↓
VERIFIED
```

A process exit code of zero is therefore not enough to make a valid ArenaPilot run.

Inspect runs with:

```bash
arena run list
arena run show run001
arena run logs run001
arena run verify run001
```

## MLflow tracking and artifact index

Training code never talks to MLflow directly. After the standard artifact contract passes, the ArenaPilot runtime logs parameters, metrics, per-fold metrics, and the run directory into MLflow, then atomically records the MLflow Run reference and local artifact index in Arena DB.

The default tracking layout is:

```text
~/.arenapilot/
├── mlflow.db
└── mlartifacts/
    └── kaggle/
        └── <competition>/
```

Set `ARENAPILOT_HOME` to move or isolate this runtime state. MLflow experiments use the naming convention `arena/<platform>/<tracking.experiment_name>`.

Arena DB schema v2 adds `artifact_refs`, which stores each verified artifact's kind, local file URI, SHA-256, and size. The first verified run for an experiment becomes its canonical run; later repeat executions remain separate evidence and do not replace it automatically.

The ownership boundary remains explicit:

```text
Arena DB
= run state, canonical run, artifact references, MLflow run ID

MLflow
= params, metrics, fold metrics, copied artifacts
```

## Compare experiments

Only canonical VERIFIED Runs in the same comparison domain can be compared directly:

```bash
arena exp compare exp001 exp002
arena exp compare exp001 exp002 --json
```

ArenaPilot checks `comparison_domain_hash` before reading metric deltas. If validation domains differ, comparison fails with `COMPARISON_DOMAIN_MISMATCH`; there is intentionally no force flag that would turn incompatible scores into a misleading numeric delta.

A successful comparison reports:

```text
primary metric: baseline -> candidate
raw delta: candidate - baseline
direction-normalized delta: positive always means improvement
fold-by-fold deltas
fold standard deviation change
runtime delta when recorded
model / pipeline / seed / runtime config changes
```

The comparison uses the Experiment's pinned canonical Run rather than automatically choosing the best repeat/seed Run. This avoids turning repeated executions into implicit leaderboard-style cherry-picking.

## Execute on Kaggle compute

ArenaPilot uses the official Kaggle CLI as a provider adapter. A frozen Experiment can be dispatched without changing its artifact or MLflow contract:

```bash
export ARENA_KAGGLE_OWNER=<kaggle-username>
arena exp run exp001 --backend kaggle
```

The default remote behavior is asynchronous. ArenaPilot creates a new Run, builds a self-contained script bundle under `.arena/bundles/<run>`, pushes it with `kaggle kernels push`, records the provider job in SQLite, and returns the Run in `queued` state.

The bundle embeds the workspace `src/` tree and materializes it in the Kaggle runtime before invoking the same entrypoint:

```bash
python -m src.train
```

The remote process receives the same Arena environment contract. `ARENA_DATA_DIR` points to `/kaggle/input/<competition-slug>` and `ARENA_OUTPUT_DIR` points to `/kaggle/working`. The competition itself is attached through `competition_sources` in `kernel-metadata.json`.

Synchronize or diagnose a remote Run with:

```bash
arena remote status run001
arena remote logs run001
arena remote recover run001
```

Recovery only proceeds after Kaggle reports the kernel complete. ArenaPilot downloads the latest kernel output, places it under `outputs/runs/<run>`, and then sends those files through the exact same local verification and MLflow-ingestion path. A Kaggle process is therefore not considered VERIFIED merely because the provider reports completion.

Remote state is intentionally split:

```text
Arena Run
created -> queued -> running -> completed -> verifying -> verified

Remote Job
created -> submitted -> queued/running -> completed/failed

Recovery
pending -> pulled -> verified/failed
```

Arena DB schema v3 adds `remote_jobs`, including the provider kernel reference, bundle path, provider state, and recovery state. Kaggle credentials remain outside ArenaPilot state. The non-secret owner name is resolved from `ARENA_KAGGLE_OWNER`, `KAGGLE_USERNAME`, or legacy `~/.kaggle/kaggle.json` metadata.

`compute.kaggle.accelerator: gpu` uses Kaggle's default GPU selection. A concrete accelerator ID in `arena.yaml` is forwarded to `kaggle kernels push --accelerator`.

## Submit verified evidence

A submission is a separate entity from its Experiment and Run. `predictions.parquet` remains experiment evidence, while `submission.csv` is the provider-ready representation that may include calibration, blending, ranking, clipping, or other submission-specific post-processing.

Create an immutable submission artifact from a VERIFIED Run:

```bash
arena submit create --run run001
# or explicitly choose a provider-ready CSV
arena submit create --run run001 --file reports/blend.csv --message "blend v1"
```

Without `--file`, ArenaPilot expects `outputs/runs/<run>/submission.csv`. It copies the exact bytes to `submissions/subXXX.csv` and stores the SHA-256 in Arena DB so later edits cannot silently change what was reviewed for submission.

Validate it against the competition sample submission:

```bash
arena submit validate sub001
# when the competition uses a nonstandard sample filename
arena submit validate sub001 --sample data/raw/gender_submission.csv
```

Validation checks exact headers, row count, first-column ID values and order, duplicate IDs, blank cells, and prediction semantics from the source Run's validation snapshot. Probability predictions must be finite and inside `[0, 1]`; value predictions must be finite.

Only a validated submission can be sent:

```bash
arena submit budget
arena submit budget --provider
arena submit send sub001 --message "exp017 frequency encoding"
```

ArenaPilot enforces `submission.daily_budget` and `submission.total_budget` from `arena.yaml` immediately before the Kaggle call. A budget of `0` means unlimited. Local daily accounting uses UTC. `--provider` additionally asks the official Kaggle CLI for the team's current provider-side submission limits.

Kaggle returns a numeric submission reference, which is stored with the exact file hash and source Run. Synchronize scoring later with:

```bash
arena submit status sub001
arena submissions --sync
```

The lifecycle is:

```text
VERIFIED Run
    ↓
submission.csv / explicit CSV
    ↓
CREATED
    ↓
sample/schema validation
    ↓
VALIDATED
    ↓
Arena budget gate
    ↓
kaggle competitions submit
    ↓
SUBMITTED / PENDING
    ↓
score synchronization
    ↓
SCORED / FAILED
```

Arena DB schema v4 adds `submissions`, preserving source Run lineage, immutable file hash, provider reference, message, public/private scores, and timestamps. Creating several submissions from the same Run is allowed; each is independent evidence of the exact representation sent to the provider.

## Workspace contract

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
    ├── bundles/
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

The next major product step after the Agent Skill contract is dogfooding the full workflow on a real competition and closing runtime gaps exposed by that run.
