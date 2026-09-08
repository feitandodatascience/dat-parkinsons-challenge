# Blend: CNN (baseline) + SBR gradient boosting

- n = 1362, features = 61, fold column = `fold_strat`

## Components, out of fold

| model | log loss | AUROC | ECE |
|---|---:|---:|---:|
| CNN | 0.3271 | 0.9302 | 0.0404 |
| SBR + trees | 0.3919 | 0.9046 | 0.0308 |

- correlation of the two logit vectors: **0.797**

A low correlation is the point. Two models that agree everywhere cannot help each
other; the blend is worth something precisely to the extent they disagree.

## Blend

Both rows below are a logistic stack scored by the same 5-fold cross-validation. They
differ only in whether the tree model's logit is offered as a second column, so the
difference measures what that model adds — not a recalibration the CNN would have got
on its own.

| stack over | parameters | cross-validated log loss |
|---|---:|---:|
| CNN logit alone | 2 | 0.3224 |
| CNN + tree logits | 3 | **0.3048** |

- **gain from adding the tree model: -0.0175** (negative is an improvement)
- same stack scored in-sample, for reference: 0.3000 (optimistic by +0.0048)

- fitted coefficients: CNN +0.603, trees +0.382 (trees carry 39% of the total weight)
- intercept +0.030

A CNN coefficient below 1.0 is the temperature scaling reappearing inside the stack —
expected, and the reason the weight is fitted this way rather than as a convex blend.
A tree coefficient near zero means the features add nothing the CNN did not already
have, whatever the tree model scores on its own.

## Verdict: worth shipping

Next: teach `submission_src/main.py` to compute these features per scan and carry
the fitted model. Refit on lightgbm rather than scikit-learn first — its text
booster format does not depend on the library version the runtime happens to have.
