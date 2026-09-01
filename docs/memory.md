# Cross-Competition Memory v1

ArenaPilot Memory separates experimental telemetry from reusable knowledge.

```text
Run / Submission
      ↓ provenance
Immutable Evidence
      ↓ interpretation
Competition-local Finding
      ↓ explicit approval
arena learn
      ↓
Versioned cross-competition Knowledge candidate
      ↓
fingerprint-aware retrieval
      ↓
Agent judgment
```

MLflow answers **what happened in a run**. Arena Memory answers **what was learned, from which evidence, and under what competition context**.

## Competition fingerprint

A fingerprint is a structured snapshot of the current competition context. ArenaPilot automatically includes task type and metric. Dataset properties are split into `observed` and `inferred` sections so measured facts are not silently mixed with judgments.

```bash
arena fingerprint set \
  --observed-json '{"dataset":{"rows_bucket":"large","high_cardinality":true},"modalities":{"tabular":true}}' \
  --inferred-json '{"structure":{"grouped":false},"shift":{"train_test_shift":"low"}}'

arena fingerprint show --json
```

Fingerprints are canonicalized and SHA-256 hashed. Re-recording identical content is idempotent.

## Evidence

Evidence is immutable and must retain provenance. It is not a conclusion.

A direct experiment comparison can be captured from the existing comparison contract:

```bash
arena evidence compare \
  --subject technique:frequency_encoding \
  --baseline exp001 \
  --candidate exp002 \
  --summary "Frequency encoding changed leakage-safe CV under the same validation domain."
```

The evidence stores the baseline/candidate Experiment and Run IDs, comparison-domain hash, direction-normalized metric effect, fold context, and config changes.

Observed failure evidence can be attached to a VERIFIED Run or Submission:

```bash
arena evidence note \
  --subject failure_mode:group_leakage \
  --outcome warning \
  --run run004 \
  --strength 1 \
  --summary "Repeated customer IDs cross random folds."
```

Competition-local evidence strength is intentionally capped at 3. Multiple competitions are counted only by the global knowledge layer, so repeated seeds or related experiments cannot masquerade as independent competitions.

## Findings

A Finding is a competition-local interpretation of one or more Evidence records.

```bash
arena finding create \
  --subject technique:frequency_encoding \
  --conclusion supported \
  --confidence medium \
  --evidence evidence001 \
  --summary "Frequency encoding produced a small but repeatable gain under leakage-safe CV."

arena finding approve finding001
```

Findings start as `candidate`. Only explicitly `approved` findings can enter cross-competition knowledge. Competition-local confidence is limited to `low` or `medium` in v1.

Contradicting evidence is linked explicitly with repeated `--contradicting` options instead of being deleted or averaged away.

## Failure-mode registry

The v1 registry contains:

```text
temporal_leakage
group_leakage
target_leakage
public_lb_overfit
train_test_shift
invalid_cv
```

Inspect it with:

```bash
arena failure list
```

Failure modes use the same Evidence → Finding → Knowledge pipeline as successful techniques.

## Learning across competitions

`arena learn` promotes approved competition Findings into `$ARENAPILOT_HOME/knowledge.db`.

```bash
arena learn
# or
arena learn --finding finding001
```

Knowledge is versioned. New competitions create a new version with `supersedes_id` pointing to the previous version; old claims are not silently rewritten. Supporting, rejected, mixed, and inconclusive source Findings remain individually traceable.

The v1 automatic confidence policy is deliberately conservative:

- one independent competition → `low`
- two or more independent competitions with at least 60% directional consistency → `medium`
- `high` is never assigned automatically in v1

A future promotion-policy slice can add stronger independence checks and explicit high-confidence approval.

## Retrieval

Retrieval first compares the current competition fingerprint with the fingerprints attached to source evidence, then combines structured similarity with confidence and evidence strength. No vector database is required in v1.

```bash
arena knowledge retrieve
arena knowledge retrieve --query frequency --limit 5
arena knowledge show technique:frequency_encoding
```

Returned results include the number of independent competitions, positive/neutral/negative counts, contradictory evidence count, fingerprint similarity, and relevance score. Memory informs the Agent; it never creates Experiments automatically.

## Storage boundary

Competition-local memory tables live in the same `.arena/arena.db` as the workspace but use an independently versioned `memory_schema_meta` sub-schema. This keeps memory migrations decoupled from the runtime control-plane schema.

Cross-competition knowledge lives at:

```text
$ARENAPILOT_HOME/knowledge.db
```

The global knowledge database has its own schema version and contains versioned knowledge items plus copied provenance references to the source competition/finding/fingerprint. It does not own Run metrics or artifacts.
