# Paired comparison: baseline_aug_rich_ens against baseline

- n = 1362 scans, matched on uid
- both sides temperature-scaled by 5-fold cross-validation within the OOF set, so
  neither is scored on a calibration fitted to it

## Scores

| run | OOF log loss | AUROC |
|---|---:|---:|
| baseline | 0.3217 | 0.9278 |
| baseline_aug_rich_ens | 0.3064 | 0.9365 |

## The paired difference

- mean difference: **-0.01538** log loss per scan (negative favours `baseline_aug_rich_ens`)
- paired standard error: **0.00528**  ->  **2.9 sigma**
- bootstrap 95% CI: **[-0.02580, -0.00501]** over 20,000 resamples
- scans improved: **66.2%** (a sign test, ignoring magnitude)

- unpaired standard error, for contrast: 0.02237 (4x larger)

That last line is why this script exists. Comparing the two headline numbers as if
they were independent would call a real difference unresolvable.

## Verdict

**`baseline_aug_rich_ens` is better, and by more than the 0.002 margin.** The
interval excludes zero and the effect clears the threshold below which the added
complexity would not be worth it. Ship it, and refit the shipped temperature on
`baseline_aug_rich_ens`'s own OOF predictions.
