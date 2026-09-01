# Validation and Experiments

## Validation is the comparison boundary

An Experiment is meaningful only relative to one validation contract. The active
validation captures split strategy, metric/direction, prediction semantics, and OOF
requirements. Once activated, treat it as immutable.

Current CLI scope supports configuring and activating the initial `val-v1`. Creation and
migration of `val-v2+` are not yet implemented as an agent-safe CLI workflow.

Never alter validation YAML or `arena.yaml` directly to change evaluation after
activation. If the evaluation contract must change, report that the current runtime needs
a new validation-version workflow rather than silently rewriting history.

## Experiment = intent

Create a new Experiment for a new hypothesis. Use lineage when the new test derives from a
prior frozen Experiment:

```bash
arena exp new --from exp001 --relation derived_from ... --json
```

Supported parent relations are:

- `derived_from`
- `ablation_of`
- `reproduction_of`
- `ensemble_of`

A parent must be frozen before derivation.

## Draft authoring

Use `arena exp configure` instead of editing `experiments/*.yaml`:

```bash
arena exp configure exp002 \
  --model-params-json '{"depth":10,"learning_rate":0.03}' \
  --pipeline-json '{"features":{"frequency_encoding":true}}' \
  --resources-json '{"accelerator":"gpu"}' \
  --backend kaggle \
  --seed 42 \
  --tag categorical \
  --tag gpu \
  --json
```

Each provided JSON object replaces that section; omitted sections are preserved. Parent
lineage and validation binding cannot be changed by configure. Once frozen, all config is
immutable; create another Experiment instead.

## Freeze

```bash
arena exp show exp002 --json
arena exp freeze exp002 --json
```

Freeze calculates a deterministic config hash and writes an immutable snapshot. A later
hash mismatch is `FROZEN_SPEC_MODIFIED`; do not repair it by overwriting the frozen file or
DB.

## Run = execution

Every execution attempt is a new Run. Retrying does not replace prior evidence. A process
exit code of zero is insufficient: the artifact verifier must close the Run as VERIFIED.

## Canonical comparison

```bash
arena exp compare exp001 exp002 --json
```

ArenaPilot compares the pinned/canonical VERIFIED Run for each Experiment rather than
selecting the best retry. This avoids cherry-picking.

If comparison domains differ, the command fails with `COMPARISON_DOMAIN_MISMATCH`.
Do not calculate a substitute delta manually. A future bridge-validation workflow should
re-run equivalent pipelines across validation versions before cross-domain interpretation.
