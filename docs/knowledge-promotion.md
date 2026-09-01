# Knowledge Promotion & Independence

ArenaPilot separates collecting memory from trusting it.

```text
Run / Submission
      ↓
Evidence
      ↓
Finding (competition-local)
      ↓ arena learn
Knowledge candidate
      ↓ independence assessment
Explicit approval
      ↓
Approved cross-competition knowledge
```

## Independence model

A different competition ID is not automatically an independent experiment. ArenaPilot therefore records an explicit evidence unit key:

```bash
arena independence set \
  --group customer-churn-family-v1 \
  --dataset-key churn-source-2026 \
  --relation same_dataset
```

Supported relations are `independent`, `derived`, `same_dataset`, and `related`. Derived competitions should share the same `--group` as their parent evidence family and identify the parent with `--parent`.

When knowledge is assessed, ArenaPilot reports both:

- `raw_competitions`: distinct competition records that contributed findings
- `independent_units`: distinct independence groups after dependency collapse

Multiple findings or related competitions inside one group get one directional vote. If a group contains equally many supporting and rejecting findings, that group is neutral.

If no explicit profile exists, ArenaPilot conservatively uses the competition ID as an implicit independent unit. Agents should replace that fallback whenever dataset or competition lineage is known.

## Technique registry

Technique knowledge uses stable semantic keys rather than free-form labels:

```bash
arena technique register frequency_encoding \
  --category categorical_encoding \
  --description "Encode category occurrence frequency without target statistics."
```

Only an active registered technique can be explicitly approved as global technique knowledge. Deprecating a technique removes it from ranked retrieval:

```bash
arena technique deprecate frequency_encoding \
  --reason "Superseded by a new canonical definition."
```

A materially different technique should receive a new key rather than silently changing the meaning of an old key.

## Assessment and approval

Inspect the latest knowledge version:

```bash
arena knowledge assess technique:frequency_encoding
```

Automatic assessment can produce only `low` or `medium`. `high` is never assigned automatically.

Explicit approval is required:

```bash
arena knowledge approve technique:frequency_encoding \
  --confidence medium \
  --reason "Two independent datasets show the same direction."
```

The v1 promotion gates are:

```text
low
  explicit approval

medium
  >= 2 independent units
  directional consistency >= 0.60

high
  >= 4 independent directional units
  directional consistency >= 0.70
  no strength >= 2 contradiction against the majority direction
  explicit approval
```

This is intentionally stricter than the candidate confidence written by `arena learn`.

## Versioning and supersession

Knowledge candidates remain immutable versions. Approving a newer version marks older approved versions of the same subject as `superseded`; their evidence is retained.

```bash
arena knowledge history technique:frequency_encoding
```

A knowledge version can also be explicitly deprecated:

```bash
arena knowledge deprecate technique:frequency_encoding \
  --reason "Validation assumptions no longer apply."
```

Deprecation is not deletion. Historical evidence and version lineage stay queryable.

## Ranked retrieval

`arena knowledge ranked` is the promotion-aware retrieval path:

```bash
arena knowledge ranked
arena knowledge ranked --query encoding --limit 5
```

Ranking separates observed and inferred fingerprint similarity and combines:

```text
core task/metric similarity
observed fingerprint similarity
inferred fingerprint similarity
approved confidence
independent evidence coverage
evidence strength
approval status
contradiction penalty
optional lexical query match
```

Approved knowledge is preferred over an unapproved newer candidate. Deprecated knowledge and deprecated techniques are excluded. This lets agents consume the strongest currently trusted claim without losing newer candidate evidence that still needs review.

## Storage boundary

The v1 Evidence/Finding/Knowledge tables are unchanged. Promotion metadata is an additive extension:

```text
competition/.arena/arena.db
├── memory v1 tables
└── competition_independence_profile

$ARENAPILOT_HOME/knowledge.db
├── knowledge v1 tables
├── competition_independence_registry
├── technique_registry
├── knowledge_assessments
└── knowledge_promotions
```

The extension has its own `promotion_schema_meta`, so existing Memory v1 databases can adopt promotion policy without rewriting historical knowledge rows.
