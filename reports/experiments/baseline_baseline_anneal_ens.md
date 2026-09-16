# Ensemble: baseline + baseline_anneal

- n = 1362 scans, matched on uid
- averaged in **logit** space, weights [0.5, 0.5]
- **uncalibrated.** Averaging logits softens them, so the temperature differs from any
  member's. Run `04_evaluate.py` for the calibrated number and `11_compare_runs.py`
  for the paired verdict. Nothing below is a decision.

## Members

| run | OOF log loss | AUROC |
|---|---:|---:|
| baseline | 0.3271 | 0.9302 |
| baseline_anneal | 0.3336 | 0.9269 |
| **baseline_baseline_anneal_ens** | **0.3220** | **0.9309** |

## Logit correlation

The number that predicts whether this pays. The SBR tree model bought -0.0175 at 0.797;
the benefit scales roughly with (1 - rho), so 0.97 predicts about a seventh of that.

| pair | Pearson r |
|---|---:|
| baseline vs baseline_anneal | **0.9740** |

## Uncalibrated change against the best member

- best member: `baseline`
- change: **-0.00509** log loss (negative favours the ensemble)

Read this against the +-0.01 CV<->leaderboard spread in `log.md`, not against zero.
A change inside that band is unresolved, not won.

## Next

```
python scripts/04_evaluate.py --run baseline_baseline_anneal_ens
python scripts/11_compare_runs.py --run-a baseline --run-b baseline_baseline_anneal_ens
```
