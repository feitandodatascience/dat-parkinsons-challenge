# Paired comparison: baseline_anneal against baseline

- n = 1362 scans, matched on uid
- both sides temperature-scaled by 5-fold cross-validation within the OOF set, so
  neither is scored on a calibration fitted to it

## Scores

| run | OOF log loss | AUROC |
|---|---:|---:|
| baseline | 0.3217 | 0.9278 |
| baseline_anneal | 0.3294 | 0.9252 |

## The paired difference

- mean difference: **+0.00765** log loss per scan (negative favours `baseline_anneal`)
- paired standard error: **0.00634**  ->  **1.2 sigma**
- bootstrap 95% CI: **[-0.00478, +0.01996]** over 20,000 resamples
- scans improved: **45.5%** (a sign test, ignoring magnitude)

- unpaired standard error, for contrast: 0.02231 (4x larger)

That last line is why this script exists. Comparing the two headline numbers as if
they were independent would call a real difference unresolvable.

## Verdict

**No detectable difference, and the point estimate favours `baseline`.**
Do not ship the change.
