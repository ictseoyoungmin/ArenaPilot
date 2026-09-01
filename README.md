# ArenaPilot

ArenaPilot is a local-first agent runtime for reproducible machine-learning competition workflows.

> Agents may forget. ArenaPilot should not.

ArenaPilot gives an AI coding agent a stable laboratory for competition work. The agent makes scientific decisions; ArenaPilot owns experiment identity, validation contracts, execution state, artifact verification, MLflow tracking, Kaggle remote jobs and submissions, and reusable cross-competition memory.

```text
AI coding agent
      ↓ scientific judgment
ArenaPilot Skill
      ↓ supported CLI
arena runtime
      ├── validation
      ├── experiments / runs
      ├── local + Kaggle execution
      ├── MLflow tracking
      ├── comparison
      ├── submissions
      └── evidence / knowledge
```

## Quick start

ArenaPilot currently requires Python 3.12+ and is installed from the repository.

```bash
git clone https://github.com/ictseoyoungmin/ArenaPilot.git
cd ArenaPilot

python -m venv .venv
source .venv/bin/activate
pip install -e .

arena version
arena contract --json
```

Keep the ArenaPilot runtime checkout separate from competition workspaces. A typical layout is:

```text
~/src/ArenaPilot/        # runtime + Skill
~/arenas/
├── titanic/             # competition workspace
├── playground-s6e9/
└── another-competition/
```

Create a workspace from a separate competitions directory:

```bash
mkdir -p ~/arenas
cd ~/arenas

arena init kaggle:titanic
cd titanic

arena status --json
arena doctor --json
```

`arena init` creates a draft workspace. It deliberately does not invent the target, metric, prediction semantics, or validation strategy.

### Data preparation today

ArenaPilot does not yet manage Kaggle competition metadata or dataset download. Put the competition files under the workspace yourself before asking an agent to build the training pipeline.

For example:

```text
~/arenas/titanic/
└── data/
    └── raw/
        ├── train.csv
        ├── test.csv
        └── gender_submission.csv
```

For Kaggle remote execution and submissions, configure the official Kaggle CLI credentials in the normal Kaggle environment. ArenaPilot wraps the official `kaggle` CLI rather than storing provider credentials itself.

## Using ArenaPilot with an AI coding agent

ArenaPilot is designed for coding agents that can read files, edit competition source code, and execute local shell commands. Codex, Claude Code, and similar shell-capable coding agents can use the same runtime contract.

The authoritative agent instructions live at:

```text
/path/to/ArenaPilot/skills/arenapilot/SKILL.md
```

Start a new agent session in the competition workspace and tell the agent to read that Skill before making ArenaPilot-managed changes. A generic prompt is:

> Use ArenaPilot to work on this Kaggle competition. First read `/path/to/ArenaPilot/skills/arenapilot/SKILL.md` and the references it requires. Use the `arena` CLI for all ArenaPilot-managed state changes. Do not write Arena DB, MLflow state, Kaggle provider state, knowledge databases, validation specs, or experiment specs directly. Start with `arena contract --json`, `arena status --json`, and `arena doctor --json`. Inspect the competition data and source code, make scientific decisions yourself, and use only runtime capabilities that actually exist.

The Skill/runtime handshake is:

```bash
arena contract --json
arena status --json
arena doctor --json
```

The current Skill requires **Agent Contract v1**. `skills/arenapilot/contract.yaml` is machine-readable, and CI checks that the public command paths declared by the Skill still exist in the runtime.

### Agent vs Runtime boundary

```text
Agent
= inspect data and code
= choose validation strategy
= form hypotheses
= implement src/train.py and feature/model code
= interpret results
= decide what to try next

ArenaPilot CLI
= supported mutation boundary

ArenaPilot Runtime
= workspace state
= validation/experiment identity
= immutable snapshots
= run lifecycle
= artifact verification
= MLflow ingestion
= Kaggle remote execution
= submission state
= evidence and knowledge persistence
```

The agent may read workspace files, logs, artifacts, and source code. It should not directly mutate `.arena/arena.db`, MLflow databases, Kaggle provider state, the global knowledge database, or Runtime-managed experiment/validation declarations.

## Typical competition workflow

A normal ArenaPilot workflow looks like this:

```text
create workspace
      ↓
inspect competition + data
      ↓
configure intake
      ↓
configure + activate validation
      ↓
retrieve relevant prior knowledge
      ↓
create hypothesis / experiment
      ↓
configure + freeze experiment
      ↓
run locally or on Kaggle
      ↓
verify artifacts + ingest MLflow
      ↓
compare experiments
      ↓
create / validate / send submission
      ↓
record evidence + finding
      ↓
learn into cross-competition knowledge
      ↓
next hypothesis
```

ArenaPilot does not try to be AutoML. The agent remains responsible for scientific judgment and training code.

## Configure the competition contract

After inspecting the competition, record the task and primary metric:

```bash
arena intake set \
  --task binary_classification \
  --target Survived \
  --metric roc_auc \
  --direction maximize
```

Configure the initial validation contract:

```bash
arena validation configure val-v1 \
  --split stratified_kfold \
  --prediction probability

arena validation activate val-v1
arena status --json
```

Only a complete, compatible validation can be activated. Activation promotes the competition from `draft` to `ready` and persists validation/spec hashes in SQLite.

Once the competition is ready, its task/target/metric contract is immutable in the current runtime. Silent evaluation-contract changes are intentionally forbidden.

## Create experiments

An Experiment is scientific intent, not an execution.

```bash
arena exp new \
  --title baseline \
  --hypothesis "A CatBoost baseline establishes the comparison floor." \
  --model-family catboost \
  --json
```

Configure the draft through the CLI rather than editing the Runtime-managed experiment YAML:

```bash
arena exp configure exp001 \
  --model-params-json '{"depth":8,"learning_rate":0.04}' \
  --pipeline-json '{"features":{"frequency_encoding":true}}' \
  --seed 42 \
  --tag baseline \
  --json
```

Review and freeze it:

```bash
arena exp show exp001 --json
arena exp freeze exp001 --json
```

A frozen Experiment is immutable. Changing it requires a new Experiment rather than rewriting history. Derived experiments can reference a frozen parent:

```bash
arena exp new \
  --from exp001 \
  --title frequency-encoding \
  --hypothesis "Frequency encoding improves high-cardinality categoricals." \
  --model-family catboost

arena exp lineage exp002 --json
```

The core invariant is:

```text
Experiment = Intent
Run        = Execution
```

Every execution creates a new Run even when the Experiment is unchanged.

## Training entrypoint and artifact contract

The workspace training entrypoint is:

```bash
python -m src.train
```

ArenaPilot supplies:

```text
ARENA_RUN_ID
ARENA_EXPERIMENT_SPEC
ARENA_VALIDATION_SPEC
ARENA_OUTPUT_DIR
ARENA_DATA_DIR
```

Training code writes to `ARENA_OUTPUT_DIR`. For a normal cross-validation run, the expected output contract is:

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

A process exit code of zero is not sufficient for success. ArenaPilot verifies required artifacts and metric semantics before a Run becomes `VERIFIED`.

```text
CREATED
  ↓
RUNNING
  ├────────→ FAILED
  ↓
COMPLETED
  ↓
VERIFYING
  ├────────→ INVALID
  ↓
MLflow ingest
  ├────────→ COMPLETED    tracking failure; retryable
  ↓
VERIFIED
```

## Run locally

Execute a frozen Experiment locally:

```bash
arena exp run exp001 --backend local --json
```

Inspect the result:

```bash
arena run list --json
arena run show run001 --json
arena run logs run001
arena run verify run001 --json
```

The first VERIFIED Run becomes the Experiment's canonical Run. Repeat runs remain separate evidence and do not automatically replace it.

## Run on Kaggle compute

ArenaPilot uses the official Kaggle CLI as a provider adapter. Set the non-secret owner name and dispatch a frozen Experiment:

```bash
export ARENA_KAGGLE_OWNER=<kaggle-username>
arena exp run exp001 --backend kaggle --json
```

Remote execution is asynchronous. ArenaPilot creates a self-contained bundle, submits a Kaggle kernel, and records the provider job in the workspace database.

```bash
arena remote status run001 --json
arena remote logs run001
arena remote recover run001 --json
```

A Kaggle kernel completing does not make the Run VERIFIED. `remote recover` pulls the output back into the workspace and sends it through the same artifact verification and MLflow ingestion path as a local Run.

## MLflow tracking

Training code does not talk to MLflow directly. ArenaPilot ingests a successfully verified artifact set after execution.

By default, Runtime tracking state lives under `ARENAPILOT_HOME` (or the default ArenaPilot home):

```text
~/.arenapilot/
├── mlflow.db
├── mlartifacts/
└── knowledge.db
```

Arena DB is the operational control plane; MLflow stores experiment telemetry and copied artifacts.

```text
Arena DB
= Experiment / Run / Submission state
= canonical Run
= artifact references
= MLflow Run ID

MLflow
= params
= metrics
= fold metrics
= copied Run artifacts
```

## Compare experiments

ArenaPilot compares only canonical VERIFIED Runs inside the same validation comparison domain:

```bash
arena exp compare exp001 exp002 --json
```

A comparison reports the primary metric delta, a direction-normalized delta, fold-level changes, fold stability, runtime delta when available, and model/pipeline/seed/runtime configuration changes.

If validation domains differ, comparison fails with:

```text
COMPARISON_DOMAIN_MISMATCH
```

There is intentionally no force flag that converts incompatible scores into a misleading numeric comparison.

## Create and send submissions

A Submission is separate from both the Experiment and the Run. `predictions.parquet` remains experiment evidence; the provider-ready CSV may contain calibration, clipping, blending, ranking, or other submission-specific transformations.

Create an immutable submission record from a VERIFIED Run:

```bash
arena submit create --run run001 --json
```

Or explicitly choose a provider-ready CSV:

```bash
arena submit create \
  --run run001 \
  --file reports/blend.csv \
  --message "blend v1" \
  --json
```

Validate it against the competition sample submission:

```bash
arena submit validate sub001 --json
# nonstandard sample filename:
arena submit validate sub001 --sample data/raw/gender_submission.csv --json
```

Check the local submission budget and optionally provider limits:

```bash
arena submit budget --json
arena submit budget --provider --json
```

Then send explicitly:

```bash
arena submit send sub001 --message "baseline submission" --json
```

Synchronize status and scores later:

```bash
arena submit status sub001 --json
arena submissions --sync --json
```

Only VERIFIED Runs can source submissions, and ArenaPilot preserves the exact submitted file hash and Run lineage.

## Cross-competition memory

ArenaPilot separates immutable facts from interpretation and reusable knowledge:

```text
Run / Submission
      ↓
Evidence
      ↓
competition-local Finding
      ↓ explicit approval
Knowledge candidate
      ↓ independence-aware assessment
promoted reusable Knowledge
```

Record a competition fingerprint while keeping observed facts separate from inferred structure:

```bash
arena fingerprint set \
  --observed-json '{"dataset":{"high_cardinality":true}}' \
  --inferred-json '{"structure":{"grouped":false}}' \
  --json
```

Create comparison-backed evidence:

```bash
arena evidence compare \
  --subject technique:frequency_encoding \
  --baseline exp001 \
  --candidate exp002 \
  --summary "Frequency encoding improved canonical CV." \
  --json
```

Interpret it locally:

```bash
arena finding create \
  --subject technique:frequency_encoding \
  --conclusion supported \
  --summary "Frequency encoding helped this competition." \
  --evidence evidence001 \
  --confidence medium \
  --json

arena finding approve finding001 --json
arena learn --finding finding001 --json
```

Retrieve prior knowledge for a new competition:

```bash
arena knowledge ranked --json
arena knowledge ranked --query encoding --json
```

Knowledge retrieval is prior context, not proof that a technique will work. Supporting and contradictory evidence are both preserved.

For dependency-aware promotion across related competitions, ArenaPilot also provides `arena independence`, `arena technique`, and `arena knowledge assess/approve/history/deprecate`.

## Core contract

ArenaPilot is built around a few invariants:

- **Experiment != Run** — an Experiment is intent; every execution creates a new Run.
- **Frozen Experiments are immutable** — retries and variations become new Runs or Experiments.
- **Validation defines comparability** — scores are directly comparable only inside a compatible comparison domain.
- **Process success != VERIFIED Run** — artifacts and metrics must pass Runtime verification.
- **Arena DB != MLflow** — Arena DB owns operational state; MLflow owns telemetry and copied artifacts.
- **Remote jobs do not write directly to local MLflow** — outputs return through ArenaPilot verification first.
- **Submission != prediction artifact** — provider-ready representations keep exact source and file lineage.
- **Memory is evidence-based** — Findings and Knowledge keep their supporting and contradicting Evidence.
- **The `arena` CLI owns Runtime mutations** — agents do not bypass it to rewrite Runtime-managed state.

## Workspace layout

```text
competition/
├── arena.yaml
├── configs/
│   ├── validation/val-v1.yaml
│   ├── models/
│   └── pipelines/
├── experiments/
├── src/
├── data/
│   ├── raw/
│   └── processed/
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

Workspace discovery walks upward from the current directory, so `arena` commands can be run from nested source/report directories inside the competition workspace.

## Current scope

ArenaPilot currently provides the Runtime contract for Kaggle competition workspaces, versioned validation, Experiment/Run lifecycle, local and Kaggle kernel execution, artifact verification, MLflow tracking, comparable Experiment analysis, submissions, and evidence-based cross-competition memory.

The following are intentionally **not** current Runtime capabilities:

- Arena-managed Kaggle competition metadata inspection or dataset download.
- validation `val-v2+` creation/migration workflows.
- generic AutoML/HPO services.
- automatic submission without an explicit `arena submit send` call.
- hosted or multi-user control plane.

Agents should report these boundaries rather than inventing commands or silently bypassing the Runtime.

## Skill references

The agent-facing contract is under `skills/arenapilot/`:

```text
skills/arenapilot/
├── SKILL.md
├── contract.yaml
├── agents/openai.yaml
└── references/
    ├── workflow.md
    ├── validation-and-experiments.md
    ├── execution-and-recovery.md
    ├── submissions.md
    ├── memory-and-knowledge.md
    └── current-scope.md
```

Use these references for the exact agent workflow and capability boundaries. `arena contract --json` is the Runtime-side handshake for that Skill.

## Development

For contributors:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest -q
arena version
arena contract --json
```
