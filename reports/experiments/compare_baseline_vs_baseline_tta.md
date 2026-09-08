# Paired comparison: baseline_tta against baseline

- n = 1362 scans, matched on uid
- both sides temperature-scaled by 5-fold cross-validation within the OOF set, so
  neither is scored on a calibration fitted to it

## Scores

| run | OOF log loss | AUROC |
|---|---:|---:|
| baseline | 0.3217 | 0.9278 |
| baseline_tta | 0.3196 | 0.9283 |

## The paired difference

- mean difference: **-0.00215** log loss per scan (negative favours `baseline_tta`)
- paired standard error: **0.00084**  ->  **2.6 sigma**
- bootstrap 95% CI: **[-0.00385, -0.00053]** over 20,000 resamples
- scans improved: **72.5%** (a sign test, ignoring magnitude)

- unpaired standard error, for contrast: 0.02243 (27x larger)

That last line is why this script exists. Comparing the two headline numbers as if
they were independent would call a real difference unresolvable.

## Verdict

**`baseline_tta` is better, and by more than the 0.002 margin.** The
interval excludes zero and the effect clears the threshold below which the added
complexity would not be worth it. Ship it, and refit the shipped temperature on
`baseline_tta`'s own OOF predictions.
