# Memory and Knowledge

ArenaPilot separates fact, interpretation, and reusable knowledge.

```text
Run / Submission
      ↓ provenance
Immutable Evidence
      ↓
Competition-local Finding
      ↓ explicit approval
Cross-competition Knowledge candidate
      ↓ dependency-aware assessment
Explicit promotion
```

## Fingerprint

A fingerprint contains task/metric plus two provenance classes:

- `observed` — facts actually measured/seen in data or competition materials.
- `inferred` — Agent interpretation such as likely grouping or train/test shift.

Keep them separate. Do not write an inference into `observed` to make retrieval score
higher.

```bash
arena fingerprint set \
  --observed-json '{"dataset":{"high_cardinality":true}}' \
  --inferred-json '{"structure":{"grouped":false}}' \
  --json
```

## Evidence

Comparison evidence is preferred when a clean baseline/candidate comparison exists:

```bash
arena evidence compare \
  --subject technique:frequency_encoding \
  --baseline exp001 \
  --candidate exp002 \
  --summary "Frequency encoding improved canonical CV." \
  --json
```

Observation evidence can attach to a VERIFIED Run or Submission:

```bash
arena evidence note \
  --subject failure_mode:group_leakage \
  --outcome warning \
  --run run004 \
  --summary "Repeated customer IDs cross random folds." \
  --json
```

Evidence is immutable. Preserve failures and contradictions.

## Finding

A Finding is a local interpretation and must reference matching Evidence:

```bash
arena finding create \
  --subject technique:frequency_encoding \
  --conclusion supported \
  --summary "Frequency encoding helped this competition." \
  --evidence evidence001 \
  --confidence medium \
  --json

arena finding approve finding001 --json
```

Findings support `supported`, `rejected`, `mixed`, and `inconclusive`. Do not create a
Finding with no evidence.

## Learn

```bash
arena learn --finding finding001 --json
```

Learning creates a new Knowledge version rather than overwriting old claims. Opposing
Findings stay visible as contradictory evidence.

## Independence

Competition IDs alone do not prove independent evidence. Declare shared lineage:

```bash
arena independence set \
  --group customer-churn-family-v1 \
  --dataset-key churn-source-2026 \
  --relation same_dataset \
  --json
```

Relations are `independent`, `derived`, `same_dataset`, and `related`. Evidence in one
group collapses to one directional vote for promotion assessment.

## Technique registry

Technique Knowledge needs a stable active registry key before global approval:

```bash
arena technique register frequency_encoding \
  --category categorical_encoding \
  --description "Replace categories with train-derived occurrence frequency." \
  --json
```

If the definition materially changes, prefer a new key. A deprecated technique is hidden
from ranked retrieval.

## Assessment and promotion

```bash
arena knowledge assess technique:frequency_encoding --json
```

Promotion gates:

- low — explicit approval.
- medium — at least 2 independent units and at least 60% directional consistency.
- high — at least 4 independent directional units, at least 70% consistency, and no
  strength>=2 contradiction against the majority direction.

High is never automatic:

```bash
arena knowledge approve technique:frequency_encoding \
  --confidence high \
  --reason "Consistent across independent datasets." \
  --json
```

Use `arena knowledge history` to inspect superseded versions. Deprecation preserves
history.

## Retrieval

Prefer the promotion-aware path:

```bash
arena knowledge ranked --json
arena knowledge ranked --query encoding --json
```

Ranking considers task/metric match, observed similarity, inferred similarity, approval
confidence, independent coverage, evidence strength, and contradiction penalties. Treat a
high score as useful prior context, not proof that the technique will work here.
