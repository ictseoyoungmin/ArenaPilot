# Current scope and safe fallbacks

This reference is deliberately explicit about gaps. Do not invent ArenaPilot commands for
capabilities that are not implemented.

## Implemented

- atomic Kaggle workspace bootstrap
- manual task/target/metric intake
- configuration/activation of the initial `val-v1`
- Experiment creation, CLI draft configuration, freeze, lineage, show/list
- local execution and artifact verification
- asynchronous Kaggle kernel dispatch/status/log/recovery
- MLflow ingestion and artifact indexing
- canonical Experiment comparison inside one comparison domain
- provider-ready submission create/validate/budget/send/status/list
- competition fingerprint, Evidence, Finding, failure registry
- cross-competition Knowledge versions and retrieval
- dependency-aware promotion, technique registry, supersession/deprecation

## Not implemented yet

### Kaggle competition metadata/data intake

There is no `arena competition inspect` or `arena data pull` yet. The skill may read data
already present in the workspace and may inspect competition information supplied by the
user, but it must not bypass ArenaPilot by invoking Kaggle provider writes directly.

If a requested workflow depends on downloading competition data, report this runtime gap
clearly. Do not claim the data was Arena-managed when it was not.

### Validation v2+ lifecycle

The current CLI configures/activates `val-v1`; it does not yet create a new immutable
validation version after readiness. If the evaluation design must materially change, do
not edit the active validation file. Record that a validation-version/bridge workflow is
required.

### Automatic HPO / AutoML

There is no HPO service. The Agent should create explicit Experiments with clear
hypotheses and lineage. Repeated blind parameter search is not a substitute for an
Experiment rationale.

### Automatic submission

ArenaPilot can send a validated Submission, but the skill must not infer an external write
merely from a good score. Submission remains an explicit workflow operation subject to
budget and user intent.

### Hosted/multi-user runtime

The runtime is local-first. Do not assume a hosted control plane, shared MLflow server,
feature store, distributed scheduler, or multi-user locking.

## Safe fallback rule

When a required capability is missing:

1. preserve current ArenaPilot state;
2. do not mutate SQLite/MLflow/provider state directly;
3. explain the missing runtime boundary;
4. perform read-only scientific analysis if it is still useful;
5. prefer adding the missing runtime capability in a separate implementation slice rather
   than embedding an ad-hoc workaround in the skill.
