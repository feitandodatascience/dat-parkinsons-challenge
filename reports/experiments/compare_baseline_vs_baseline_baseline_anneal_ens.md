# Paired comparison: baseline_baseline_anneal_ens against baseline

- n = 1362 scans, matched on uid
- both sides temperature-scaled by 5-fold cross-validation within the OOF set, so
  neither is scored on a calibration fitted to it

## Scores

| run | OOF log loss | AUROC |
|---|---:|---:|
| baseline | 0.3217 | 0.9278 |
| baseline_baseline_anneal_ens | 0.3194 | 0.9287 |

## The paired difference

- mean difference: **-0.00234** log loss per scan (negative favours `baseline_baseline_anneal_ens`)
- paired standard error: **0.00316**  ->  **0.7 sigma**
- bootstrap 95% CI: **[-0.00851, +0.00377]** over 20,000 resamples
- scans improved: **58.1%** (a sign test, ignoring magnitude)

- unpaired standard error, for contrast: 0.02237 (7x larger)

That last line is why this script exists. Comparing the two headline numbers as if
they were independent would call a real difference unresolvable.

## Verdict

**Direction favours `baseline_baseline_anneal_ens`, but the interval crosses zero.** The gain is
not distinguishable from no change on this data. Whether to ship it comes down to
cost: if the change is already built and adds no failure modes, a small unproven
gain in the right direction is a defensible bet. If it adds inference complexity,
it is not.
