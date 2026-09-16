# Ensemble: baseline_tta + aug_rich_tta

- n = 1362 scans, matched on uid
- averaged in **logit** space, weights [0.5, 0.5]
- **uncalibrated.** Averaging logits softens them, so the temperature differs from any
  member's. Run `04_evaluate.py` for the calibrated number and `11_compare_runs.py`
  for the paired verdict. Nothing below is a decision.

## Members

| run | OOF log loss | AUROC |
|---|---:|---:|
| baseline_tta | 0.3239 | 0.9306 |
| aug_rich_tta | 0.3317 | 0.9303 |
| **ens_tta** | **0.3021** | **0.9393** |

## Logit correlation

The number that predicts whether this pays. The SBR tree model bought -0.0175 at 0.797;
the benefit scales roughly with (1 - rho), so 0.97 predicts about a seventh of that.

| pair | Pearson r |
|---|---:|
| baseline_tta vs aug_rich_tta | **0.9168** |

## Uncalibrated change against the best member

- best member: `baseline_tta`
- change: **-0.02179** log loss (negative favours the ensemble)

Read this against the +-0.01 CV<->leaderboard spread in `log.md`, not against zero.
A change inside that band is unresolved, not won.

## Next

```
python scripts/04_evaluate.py --run ens_tta
python scripts/11_compare_runs.py --run-a baseline_tta --run-b ens_tta
```
