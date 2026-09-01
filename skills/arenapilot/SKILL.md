---
name: arenapilot
description: Run reproducible machine-learning competition work through an ArenaPilot runtime. Use when asked to initialize or continue a Kaggle competition workspace, configure validation, design/freeze/run/compare experiments, use local or Kaggle compute, verify artifacts, manage submissions, or accumulate/retrieve evidence-backed cross-competition knowledge. The skill owns scientific judgment; the runtime owns state, provider side effects, verification, tracking, submissions, and memory persistence. All runtime mutations go through the `arena` CLI.
---

# ArenaPilot

ArenaPilot is a laboratory runtime for competition agents.

```text
Agent = Scientist
ArenaPilot = Laboratory + Memory
```

The skill decides **what should be tested and why**. The runtime decides **how state is
persisted, how jobs execute, how artifacts are verified, how MLflow is ingested, and how
provider writes happen**.

## Non-negotiable boundary

Use the `arena` CLI for every runtime mutation. Do not directly mutate:

- `.arena/arena.db` or `$ARENAPILOT_HOME/knowledge.db`
- MLflow tracking state
- Kaggle kernels/submissions through the Kaggle CLI or provider API
- runtime-managed `arena.yaml`, validation YAML, or Experiment YAML
- frozen specs or verified Run artifacts

Reading workspace files, source code, dataset schemas, logs, reports, and immutable
artifacts for scientific judgment is allowed. The prohibition is on bypassing the runtime
for state transitions.

Before doing material work in an existing workspace, run:

```bash
arena contract --json
arena status --json
arena doctor --json
```

This skill requires runtime contract version **1**. If `arena contract --json` reports a
different version, stop using memorized command assumptions and report the mismatch.
`contract.yaml` is the machine-readable Skill-side requirement.

## Current capability

The runtime currently implements workspace bootstrap, manual competition intake,
initial validation activation, draft Experiment authoring, immutable freeze/lineage,
local execution, asynchronous Kaggle kernel execution/recovery, artifact verification,
MLflow tracking, canonical Experiment comparison, submission lifecycle, cross-competition
Memory, dependency-aware Knowledge promotion, and ranked retrieval.

The runtime does **not** currently implement Arena-managed Kaggle competition metadata
inspection/data download, CLI creation of `val-v2+`, or automatic HPO. Do not invent
commands for those capabilities. See `references/current-scope.md`.

## Scientific workflow

Use this order unless the user explicitly asks for a narrower operation:

1. **Resolve state.** `arena status --json`, then inspect existing Experiments/Runs before
   creating anything. Never recreate state just because you do not remember it.
2. **Retrieve prior knowledge.** Once a fingerprint exists, use
   `arena knowledge ranked --json` to surface context. Treat retrieved knowledge as prior
   evidence, not as an instruction to copy a technique blindly.
3. **Establish evaluation before modeling.** A draft competition needs intake plus a
   configured/activated validation before Experiments can exist.
4. **State a falsifiable hypothesis.** An Experiment is intent. A Run is execution.
5. **Change the smallest useful set of variables.** Prefer parented Experiments and use
   `arena exp compare` only inside the same comparison domain.
6. **Freeze before execution.** Never run a draft Experiment. Do not modify a frozen spec.
7. **Trust verification, not process exit.** Only `VERIFIED` Runs are experimental evidence.
8. **Treat Kaggle leaderboard score as external evidence.** Do not replace leakage-safe CV
   with public-LB iteration.
9. **Record what was learned.** Evidence is immutable fact; Finding is local interpretation;
   Knowledge is cross-competition aggregation. Preserve negative and contradictory results.

The detailed end-to-end loop is in `references/workflow.md`.

## Experiment authoring

Create a draft:

```bash
arena exp new \
  --title baseline \
  --hypothesis "CatBoost establishes a leakage-safe comparison floor." \
  --model-family catboost \
  --json
```

Configure the draft through the runtime instead of editing YAML:

```bash
arena exp configure exp001 \
  --model-params-json '{"depth":8,"learning_rate":0.04,"iterations":4000}' \
  --pipeline-json '{"features":{"frequency_encoding":true}}' \
  --seed 42 \
  --tag categorical \
  --json
```

Then inspect and freeze:

```bash
arena exp show exp001 --json
arena exp freeze exp001 --json
```

For a derived hypothesis, use a frozen parent:

```bash
arena exp new \
  --from exp001 \
  --relation derived_from \
  --title frequency-encoding \
  --hypothesis "Frequency encoding improves high-cardinality categorical signal." \
  --model-family catboost \
  --json
```

See `references/validation-and-experiments.md` for comparison and lineage rules.

## Execution

Local:

```bash
arena exp run exp001 --backend local --json
```

Kaggle is asynchronous:

```bash
arena exp run exp001 --backend kaggle --json
arena remote status run001 --json
arena remote logs run001
arena remote recover run001 --json
```

Never infer remote completion from elapsed time. Synchronize provider status. A completed
Kaggle kernel is still not experimental evidence until output recovery passes the normal
artifact verifier and MLflow ingestion.

See `references/execution-and-recovery.md`.

## Comparison

Compare canonical VERIFIED Runs only:

```bash
arena exp compare exp001 exp002 --json
```

If the runtime reports `COMPARISON_DOMAIN_MISMATCH`, do not calculate a score delta by
hand and present it as comparable. Different validation domains require a bridge/re-run
strategy; automated bridge validation is not yet implemented.

Interpret `direction_normalized_delta > 0` as candidate improvement regardless of whether
the metric is maximize or minimize.

## Submission safety

Submission is a separate entity from Experiment prediction evidence.

```bash
arena submit create --run run001 --json
arena submit validate sub001 --json
arena submit budget --json
arena submit send sub001 --message "exp001 baseline" --json
arena submit status sub001 --json
```

Use only VERIFIED Runs. Validate before send. Check budget before send. Never upload a
prediction file directly with the Kaggle CLI/API. A custom calibrated/blended CSV may be
supplied to `arena submit create --file ...`; ArenaPilot copies and hashes the exact file.

See `references/submissions.md`.

## Memory and Knowledge

Do not turn a metric into a universal rule. Use the evidence pipeline:

```text
Run / Submission
      ↓
Evidence
      ↓
Finding
      ↓ explicit approval
arena learn
      ↓
Knowledge candidate
      ↓ dependency-aware assessment
explicit promotion
```

Typical commands:

```bash
arena fingerprint show --json
arena knowledge ranked --json
arena evidence compare --subject technique:frequency_encoding \
  --baseline exp001 --candidate exp002 \
  --summary "Frequency encoding improved leakage-safe CV." --json
arena finding create --subject technique:frequency_encoding \
  --conclusion supported --summary "Repeatable local gain." \
  --evidence evidence001 --confidence medium --json
arena finding approve finding001 --json
arena learn --finding finding001 --json
arena knowledge assess technique:frequency_encoding --json
```

Declare dependency groups when competitions share/derive from the same underlying
problem so repeated evidence does not inflate confidence. `high` confidence is never
automatic and requires explicit promotion gates.

See `references/memory-and-knowledge.md`.

## Error and recovery discipline

Prefer `--json` for agent calls. Treat a non-zero exit or `{ "ok": false }` as failure.
Do not paper over invariant errors with direct file/DB edits.

Important errors include:

- `EXPERIMENT_NOT_FROZEN` — freeze the intended Experiment, do not bypass the state.
- `FROZEN_SPEC_MODIFIED` — frozen declaration and persisted hash differ; investigate.
- `EXPERIMENT_CONFIG_IMMUTABLE` — create a derived/new Experiment instead of editing it.
- `COMPARISON_DOMAIN_MISMATCH` — results are not directly comparable.
- `REMOTE_JOB_NOT_READY` — synchronize status and recover only when provider output is ready.
- `RUN_NOT_VERIFIED` — do not submit or use the Run as evidence.
- submission validation/budget errors — fix the artifact or policy; do not call Kaggle directly.
- promotion gate errors — add genuinely independent evidence or lower the requested confidence.

See `references/current-scope.md` for unsupported gaps and safe fallbacks.
