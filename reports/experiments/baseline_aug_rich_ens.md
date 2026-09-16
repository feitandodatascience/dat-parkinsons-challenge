# Ensemble: baseline + aug_rich

- n = 1362 scans, matched on uid
- averaged in **logit** space, weights [0.5, 0.5]
- **uncalibrated.** Averaging logits softens them, so the temperature differs from any
  member's. Run `04_evaluate.py` for the calibrated number and `11_compare_runs.py`
  for the paired verdict. Nothing below is a decision.

## Members

| run | OOF log loss | AUROC |
|---|---:|---:|
| baseline | 0.3271 | 0.9302 |
| aug_rich | 0.3358 | 0.9288 |
| **baseline_aug_rich_ens** | **0.3044** | **0.9385** |

## Logit correlation

The number that predicts whether this pays. The SBR tree model bought -0.0175 at 0.797;
the benefit scales roughly with (1 - rho), so 0.97 predicts about a seventh of that.

| pair | Pearson r |
|---|---:|
| baseline vs aug_rich | **0.9135** |

## Uncalibrated change against the best member

- best member: `baseline`
- change: **-0.02269** log loss (negative favours the ensemble)

Read this against the +-0.01 CV<->leaderboard spread in `log.md`, not against zero.
A change inside that band is unresolved, not won.

## Next

```
python scripts/04_evaluate.py --run baseline_aug_rich_ens
python scripts/11_compare_runs.py --run-a baseline --run-b baseline_aug_rich_ens
```
