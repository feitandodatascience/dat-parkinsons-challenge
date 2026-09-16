# Blend: CNN (baseline_tta) + SBR gradient boosting

- n = 1362, features = 157, fold column = `fold_strat`
- components available to the stack: 3
- scores clipped at 1e-06; bootstrap over 10,000 resamples

## Components, out of fold

| model | log loss | AUROC | ECE |
|---|---:|---:|---:|
| CNN (baseline_tta) | 0.3239 | 0.9306 | 0.0370 |
| SBR trees (default) | 0.3659 | 0.9184 | 0.0329 |
| SBR trees (tuned) | 0.3617 | 0.9202 | 0.0302 |

### Pairwise logit correlations

| a | b | rho | 1 - rho |
|---|---|---:|---:|
| CNN (baseline_tta) | SBR trees (default) | 0.825 | 0.175 |
| CNN (baseline_tta) | SBR trees (tuned) | 0.810 | 0.190 |
| SBR trees (default) | SBR trees (tuned) | 0.980 | 0.020 |

A low correlation is the point. Two models that agree everywhere cannot help each
other; a component is worth something precisely to the extent it disagrees. The
governing measurement: the trees bought -0.0175 at rho = 0.80, and a second CNN run
bought -0.0022 at rho = 0.974 — a ratio that (1 - rho) predicted in advance. Read the
last column before reading the coefficients.

## Variants

Every row is a logistic stack — one coefficient per component plus an intercept —
scored by the same 5-fold cross-validation. Rows differ only in which
columns were offered, so a difference measures information added, not a recalibration
the CNN would have got on its own.

The paired difference is against **CNN + trees (default)**, the shipped configuration.
Negative favours the variant. Both sides are already stack-calibrated out of fold, so
unlike `11_compare_runs.py` no temperature is applied here — that would apply one twice.

The score column is the **cross-validated log loss** of that stack; it is held out at
every level, so it is not the minimum of a curve fitted to the rows it is quoted on.

| stack over | params | CV log loss | Δ vs shipped | σ | 95% CI | scans improved |
|---|---:|---:|---:|---:|---|---:|
| CNN alone | 2 | 0.3201 | +0.02296 | 3.9σ | [+0.01163, +0.03443] | 35.7% |
| CNN + trees (default) | 3 | 0.2972 | — | — | — (reference) | — |
| CNN + trees (tuned) **<- ships** | 3 | 0.2929 | -0.00429 | 2.1σ | [-0.00829, -0.00022] | 57.8% |

- **gain from adding the tree model: -0.0273** (negative is an improvement)
- same stack scored in-sample, for reference: 0.2893 (optimistic by +0.0036)

- fitted coefficients: CNN +0.560, trees +0.422 (trees carry 43% of the total weight)
- intercept +0.025

A CNN coefficient below 1.0 is the temperature scaling reappearing inside the stack —
expected, and the reason the weight is fitted this way rather than as a convex blend.
A coefficient near zero means that component adds nothing the CNN did not already
have, whatever it scores on its own.

**A paired difference smaller than about 0.01 is not resolved by this OOF set**, even
when the interval excludes zero: the observed CV-to-leaderboard gap has run from
+0.0024 to -0.0088. Read σ and the interval, never the point estimate alone.

## Which tree configuration ships

Fixed in advance, not chosen by reading the table: candidates are ordered by increasing
complexity and promoted only on a mean better than -0.002 **and** a bootstrap
interval excluding zero. Each comparison is against the incumbent at that point, not
against the default, so a chain of near-ties cannot accumulate into a promotion.

| candidate | against | Δ | σ | 95% CI | improved | promoted |
|---|---|---:|---:|---|---:|---|
| trees (tuned) | trees (default) | -0.00429 | 2.1σ | [-0.00829, -0.00022] | 57.8% | **yes** |

- shipping: **trees (tuned)**

## Seed bagging

Not run (`--gbm-bag 1`, the default). One booster per fold, exactly as before.

Boosted trees at this n are a high-variance estimator and log loss pays directly for
variance reduction — the entire measured value of TTA in this project (-0.0022) was
variance reduction rather than better ranking. `--gbm-bag S` fits S boosters per
fold under fixed distinct seeds and averages them in logit space inside the fold.

## Hyperparameter tuning

- grid: 12 configurations
- selection: **nested** — inner 3-fold cross-validation within each outer fold's training rows
- the held-out fold is never seen by the selection, so **the CV log loss reported
  above is NOT optimistic on account of the tuning.** The choice is part of the
  procedure being cross-validated, exactly as the stack's own fit is.

Configurations selected across the outer folds:

| configuration | folds that chose it |
|---|---:|
| `deeper-regularised` | 3 |
| `high-min-child` | 1 |
| `stumps-slow-long` | 1 |

Folds may disagree, and that is not a bug — it is the honest version. The shipped
boosters each carry the parameters chosen on their own training rows, so the shipped
ensemble is bit-for-bit the procedure that was measured. Refitting one 'best'
configuration on all the data would ship something that was never scored.

One residual optimism that no nesting can remove: the **decision** to prefer a
tuned candidate is taken by looking at held-out numbers, so the margin by which it
wins is itself selected. The acceptance chain above is the mitigation — a rule fixed
in advance rather than 'whichever number is lower' — but a promoted candidate's
quoted gain should still be read as the optimistic end of its interval.

## Clip epsilon

Swept on the held-out predictions of the configuration that would ship. The equivalent
sweep for the CNN alone is in `reports/experiments/baseline.md`.

| clip eps | CV log loss |
|---:|---:|
| 1e-06 | 0.2929 **<- best** |
| 1e-05 | 0.2929 |
| 0.0001 | 0.2929 |
| 0.001 | 0.2929 |
| 0.005 | 0.2930 |
| 0.01 | 0.2936 |
| 0.02 | 0.2957 |
| 0.05 | 0.3027 |

- best on this OOF set: **1e-06** (0.2929), against
  0.2929 at the shipped 1e-06 — a difference of
  +0.0000.
- **this minimum is chosen in-sample.** Any apparent gain from moving off the shipped
  epsilon is optimistic by exactly the amount the curve was fitted to this data, so a
  tie to four decimals is a reason to keep the incumbent, not to switch.
- a tight clip is insurance, not accuracy: it caps what one confidently-wrong scan can
  cost. On a test set drawn differently from this one, the flat region is the safe place
  to sit.

## Verdict: worth shipping

Artifacts are written next to the predictions that justify them, in
`models/baseline_tta_blend/`: `gbm_fold*.txt` (lightgbm boosters, text
format, version-portable) and `stack.json` (the three coefficients plus the feature
order). To build a submission carrying them:

```
python scripts/05_package_submission.py --config conf/baseline.yaml --blend-from baseline_tta_blend --fallback 0.5485 --clip-eps 1e-06
```

`--temperature` is neither needed nor accepted alongside `--blend-from`: the stack's
own CNN coefficient absorbs the calibration, so passing a temperature as well would
apply it twice.

## Verify it

The held-out predictions of the shipped configuration are banked as a run of their own,
so the improvement can be given a standard error instead of being read off the pooled
numbers above:

```bash
python scripts/11_compare_runs.py --run-a baseline_tta --run-b baseline_tta_blend
```
