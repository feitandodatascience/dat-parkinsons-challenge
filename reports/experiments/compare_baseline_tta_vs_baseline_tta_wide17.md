# Paired comparison: baseline_tta_wide17 against baseline_tta

- n = 1362 scans, matched on uid
- both sides temperature-scaled by 5-fold cross-validation within the OOF set, so
  neither is scored on a calibration fitted to it

## Scores

| run | OOF log loss | AUROC |
|---|---:|---:|
| baseline_tta | 0.3196 | 0.9283 |
| baseline_tta_wide17 | 0.3195 | 0.9282 |

## The paired difference

- mean difference: **-0.00004** log loss per scan (negative favours `baseline_tta_wide17`)
- paired standard error: **0.00021**  ->  **0.2 sigma**
- bootstrap 95% CI: **[-0.00047, +0.00037]** over 20,000 resamples
- scans improved: **55.4%** (a sign test, ignoring magnitude)

- unpaired standard error, for contrast: 0.02242 (106x larger)

That last line is why this script exists. Comparing the two headline numbers as if
they were independent would call a real difference unresolvable.

## Verdict

**Direction favours `baseline_tta_wide17`, but the interval crosses zero.** The gain is
not distinguishable from no change on this data. Whether to ship it comes down to
cost: if the change is already built and adds no failure modes, a small unproven
gain in the right direction is a defensible bet. If it adds inference complexity,
it is not.
