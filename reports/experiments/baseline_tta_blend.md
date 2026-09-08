# Blend: CNN (baseline_tta) + SBR gradient boosting

- n = 1362, features = 61, fold column = `fold_strat`

## Components, out of fold

| model | log loss | AUROC | ECE |
|---|---:|---:|---:|
| CNN | 0.3239 | 0.9306 | 0.0370 |
| SBR + trees | 0.3920 | 0.9042 | 0.0242 |

- correlation of the two logit vectors: **0.798**

A low correlation is the point. Two models that agree everywhere cannot help each
other; the blend is worth something precisely to the extent they disagree.

## Blend

Both rows below are a logistic stack scored by the same 5-fold cross-validation. They
differ only in whether the tree model's logit is offered as a second column, so the
difference measures what that model adds — not a recalibration the CNN would have got
on its own.

| stack over | parameters | cross-validated log loss |
|---|---:|---:|
| CNN logit alone | 2 | 0.3201 |
| CNN + tree logits | 3 | **0.3038** |

- **gain from adding the tree model: -0.0163** (negative is an improvement)
- same stack scored in-sample, for reference: 0.2983 (optimistic by +0.0055)

- fitted coefficients: CNN +0.616, trees +0.378 (trees carry 38% of the total weight)
- intercept +0.024

A CNN coefficient below 1.0 is the temperature scaling reappearing inside the stack —
expected, and the reason the weight is fitted this way rather than as a convex blend.
A tree coefficient near zero means the features add nothing the CNN did not already
have, whatever the tree model scores on its own.

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

The held-out blended predictions are banked as a run of their own, so the improvement
can be given a standard error instead of being read off the pooled numbers above:

```bash
python scripts/11_compare_runs.py --run-a baseline_tta --run-b baseline_tta_blend
```
