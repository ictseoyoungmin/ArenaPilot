# Execution and recovery

## Local execution

```bash
arena exp run exp001 --backend local --json
```

The runtime creates a new Run, launches `python -m src.train`, captures logs, validates the
standard artifacts, ingests verified evidence into MLflow, and persists artifact hashes.

Do not call the training module directly when the goal is an ArenaPilot Run; doing so
bypasses Run identity, verification, tracking, and lineage.

## Run states

The important terminal distinctions are:

- `verified` — artifact contract and tracking ingestion succeeded; usable evidence.
- `invalid` — execution completed but the output contract failed verification.
- `failed` — execution process/provider failed.

A successful subprocess is not automatically a verified Run.

Inspect with:

```bash
arena run show run001 --json
arena run logs run001
arena run verify run001 --json
```

`run verify` is appropriate for a completed/invalid Run after its expected artifacts have
been repaired through the normal workspace workflow. Never mutate DB status directly.

## Kaggle remote execution

Remote execution is intentionally asynchronous:

```bash
export ARENA_KAGGLE_OWNER=<username>
arena exp run exp001 --backend kaggle --json
```

The runtime packages a self-contained script bundle and submits it through the official
Kaggle provider adapter. The command returns a queued Run instead of blocking for training.

Synchronize state explicitly:

```bash
arena remote status run001 --json
arena remote logs run001
```

When the provider is complete:

```bash
arena remote recover run001 --json
```

Recovery pulls output into the local Run directory and then uses the same verifier +
MLflow ingest path as local execution. Never treat Kaggle `complete` as VERIFIED before
recovery succeeds.

## Recovery rules

- `REMOTE_JOB_NOT_READY`: synchronize later; do not fabricate completion.
- Provider failure: inspect `arena remote logs`; create a new Run for another execution
  attempt rather than rewriting the failed Run.
- Verification failure after output pull: inspect local Run logs/artifacts, preserve the
  invalid evidence, then make a new Experiment/Run or re-verify only when appropriate.
- Tracking failure after valid execution: the runtime is designed to make verification
  retryable; do not write to MLflow directly.

Remote kernels must never connect to local MLflow directly. The evidence crosses the
provider boundary as files, is verified locally, then is ingested by ArenaPilot.
