# ArenaPilot workflow

Use this as the default control loop for one competition.

## 1. Resolve before acting

From inside the workspace:

```bash
arena contract --json
arena status --json
arena doctor --json
arena exp list --json
arena run list --json
arena submissions --json
```

Do not create a replacement Experiment or Run merely because the current conversation has
forgotten one. ArenaPilot state is the source of continuity.

## 2. Retrieve prior context

If a competition fingerprint exists:

```bash
arena fingerprint show --json
arena independence show --json
arena knowledge ranked --json
```

Use retrieved Knowledge to propose hypotheses, not to skip validation. Pay attention to
contradictions, independent evidence units, applicability, and observed-vs-inferred
fingerprint match.

## 3. Make the competition ready

A fresh workspace is draft. Configure only facts you actually know:

```bash
arena intake set \
  --task binary_classification \
  --target target \
  --metric roc_auc \
  --direction maximize \
  --json

arena validation configure val-v1 \
  --split stratified_kfold \
  --prediction probability \
  --json

arena validation activate val-v1 --json
```

Do not invent target, metric, grouping, time semantics, or prediction semantics. If these
facts are unknown, inspect the competition/data first. Arena-managed Kaggle metadata/data
intake is not implemented yet.

## 4. Create one hypothesis per Experiment

```bash
arena exp new \
  --title baseline \
  --hypothesis "A simple baseline establishes the comparison floor." \
  --model-family catboost \
  --json

arena exp configure exp001 \
  --model-params-json '{"depth":8}' \
  --pipeline-json '{"features":{}}' \
  --seed 42 \
  --json

arena exp show exp001 --json
arena exp freeze exp001 --json
```

For a changed hypothesis, create a new Experiment with `--from`; do not repurpose a
frozen Experiment.

## 5. Execute and verify

Local:

```bash
arena exp run exp001 --backend local --json
```

Kaggle:

```bash
arena exp run exp001 --backend kaggle --json
arena remote status run001 --json
arena remote recover run001 --json
```

Only `VERIFIED` Runs are evidence. `FAILED` means execution failure. `INVALID` means the
process may have completed but the artifact contract did not verify.

## 6. Compare before interpreting

```bash
arena exp compare exp001 exp002 --json
```

Use canonical VERIFIED Runs. Positive `direction_normalized_delta` means the candidate is
better. Also inspect fold deltas/stability and the declared config changes. Never directly
compare mismatched validation domains.

## 7. Submit deliberately

```bash
arena submit create --run run002 --json
arena submit validate sub001 --json
arena submit budget --json
arena submit send sub001 --message "exp002" --json
arena submit status sub001 --json
```

Leaderboard results are external evidence, not a replacement for the validation contract.

## 8. Record evidence and learning

```bash
arena evidence compare \
  --subject technique:frequency_encoding \
  --baseline exp001 \
  --candidate exp002 \
  --summary "Frequency encoding improved the canonical CV result." \
  --json

arena finding create \
  --subject technique:frequency_encoding \
  --conclusion supported \
  --summary "Frequency encoding helped in this competition." \
  --evidence evidence001 \
  --confidence medium \
  --json

arena finding approve finding001 --json
arena learn --finding finding001 --json
arena knowledge assess technique:frequency_encoding --json
```

Register a technique before approving technique Knowledge globally. Declare independence
relationships before asking for high confidence.
