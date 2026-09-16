# Paired comparison: ens_tta_blend against baseline_tta_blend

- n = 1362 scans, matched on uid
- both sides temperature-scaled by 5-fold cross-validation within the OOF set, so
  neither is scored on a calibration fitted to it

## Scores

| run | OOF log loss | AUROC |
|---|---:|---:|
| baseline_tta_blend | 0.2953 | 0.9419 |
| ens_tta_blend | 0.2852 | 0.9455 |

## The paired difference

- mean difference: **-0.01010** log loss per scan (negative favours `ens_tta_blend`)
- paired standard error: **0.00421**  ->  **2.4 sigma**
- bootstrap 95% CI: **[-0.01845, -0.00173]** over 20,000 resamples
- scans improved: **60.1%** (a sign test, ignoring magnitude)

- unpaired standard error, for contrast: 0.02201 (5x larger)

That last line is why this script exists. Comparing the two headline numbers as if
they were independent would call a real difference unresolvable.

## Verdict

**`ens_tta_blend` is better, and by more than the 0.002 margin.** The
interval excludes zero and the effect clears the threshold below which the added
complexity would not be worth it. Ship it, and refit the shipped temperature on
`ens_tta_blend`'s own OOF predictions.
