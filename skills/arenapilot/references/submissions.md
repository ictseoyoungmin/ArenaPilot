# Submissions

Submission is a provider-facing artifact derived from a VERIFIED Run. It is not the same
object as an Experiment or its prediction evidence.

## Create

Default provider-ready file:

```bash
arena submit create --run run001 --json
```

Custom post-processed file:

```bash
arena submit create \
  --run run001 \
  --file reports/calibrated.csv \
  --message "calibration v1" \
  --json
```

ArenaPilot copies the exact CSV into `submissions/subXXX.csv` and fixes its SHA-256. This
allows raw, calibrated, and blended submissions from the same source Run without mutating
`predictions.parquet`.

## Validate

```bash
arena submit validate sub001 --json
```

By default validation uses `data/raw/sample_submission.csv`. It checks schema, row count,
ID/order alignment, duplicate IDs, blanks, and prediction semantics such as probability
range when applicable.

Do not send a submission that has not passed runtime validation.

## Budget

```bash
arena submit budget --json
arena submit budget --provider --json
```

The first command checks ArenaPilot's local daily/total policy. `--provider` additionally
queries Kaggle limits. Never bypass a local budget failure by calling Kaggle directly.

## Send and synchronize

```bash
arena submit send sub001 --message "exp001 baseline" --json
arena submit status sub001 --json
arena submissions --sync --json
```

`send` is the explicit provider write. The skill should not autonomously infer that a
validated file must be submitted; send when the user's requested workflow calls for an
actual competition submission and policy allows it.

Public/private leaderboard scores are external evidence. Do not use public leaderboard
feedback to retroactively redefine the validation contract or cherry-pick Runs.
