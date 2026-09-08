# Paired comparison: baseline_tta_blend against baseline

- n = 1362 scans, matched on uid
- both sides temperature-scaled by 5-fold cross-validation within the OOF set, so
  neither is scored on a calibration fitted to it

## Scores

| run | OOF log loss | AUROC |
|---|---:|---:|
| baseline | 0.3217 | 0.9278 |
| baseline_tta_blend | 0.3061 | 0.9375 |

## The paired difference

- mean difference: **-0.01562** log loss per scan (negative favours `baseline_tta_blend`)
- paired standard error: **0.00503**  ->  **3.1 sigma**
- bootstrap 95% CI: **[-0.02551, -0.00566]** over 20,000 resamples
- scans improved: **63.4%** (a sign test, ignoring magnitude)

- unpaired standard error, for contrast: 0.02233 (4x larger)

That last line is why this script exists. Comparing the two headline numbers as if
they were independent would call a real difference unresolvable.

## Verdict

**`baseline_tta_blend` is better, and by more than the 0.002 margin.** The
interval excludes zero and the effect clears the threshold below which the added
complexity would not be worth it. Ship it, and refit the shipped temperature on
`baseline_tta_blend`'s own OOF predictions.
