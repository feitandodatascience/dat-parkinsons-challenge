# Vetted aggregate summaries

Files here are readable by Claude Code. Everything in this directory must be **aggregate and
non-identifying**.

## How things get here

1. Run an analysis script (e.g. `python scripts/01_inspect_data.py`).
2. It writes to `reports/eda/raw/`, which is quarantined — Claude cannot read it.
3. **You** read the draft, confirm it contains no per-sample information, and copy it here.

That vetting step is a human judgement and is the whole point. Do not automate it.

## Allowed

- Counts, class balance, positive rates
- Distributions over the full dataset: quantiles, min/max, histograms
- Distinct shape/spacing values and their frequencies
- Per-group aggregates where the groups are large
- Metric values, loss curves, timings

## Not allowed

- Any `uid`
- Any per-sample row, value, or voxel
- Images or renders of scans
- Statistics over a group small enough to identify an individual

Test to apply: *could this be inverted to recover one patient's scan or label?*
If yes — or if you are unsure — it stays in `raw/`.
