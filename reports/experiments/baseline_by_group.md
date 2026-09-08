# Error decomposition by acquisition family: baseline

Computed from existing out-of-fold predictions — no training was run.
These folds are STRATIFIED, so the model saw every family during training. Read the
AUROC column as a real model property and the calibration columns as a lower bound on
what grouped CV will show.

- n = 1362, positive rate = 0.548
- distinct acquisition families = 21 (6 at or above the n >= 30 reporting threshold)
- global temperature fitted on all OOF = 1.2614

## Per-family breakdown

`prior floor` is the log loss of predicting that family's own base rate — the score to
beat *within* the family. `oracle shift` is the best log loss reachable by adding one
constant to the logits, chosen with the family's own labels; the gap between it and
`logloss` is the part of the family's error that is a systematic offset rather than bad
ranking. It is an oracle and therefore a bound, not an achievable score.

| family | n | pos rate | mean pred | logloss | prior floor | oracle shift | offset | AUROC | ECE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 256@2.5x2.5x2.5mm | 460 | 59.3% | 0.570 | 0.3691 | 0.6756 | 0.3659 | +0.27 | 0.9176 | 0.0509 |
| 128@3.9x3.9x3.9mm | 325 | 58.2% | 0.574 | 0.3342 | 0.6798 | 0.3338 | +0.10 | 0.9277 | 0.0544 |
| 128@2.3x2.3x2.3mm | 207 | 50.2% | 0.479 | 0.2768 | 0.6931 | 0.2733 | +0.29 | 0.9396 | 0.0397 |
| 128@2.4x2.4x2.4mm | 149 | 46.3% | 0.486 | 0.2339 | 0.6904 | 0.2290 | -0.44 | 0.9560 | 0.0407 |
| small families (pooled) | 125 | 48.0% | 0.470 | 0.3580 | 0.6923 | 0.3573 | +0.15 | 0.9097 | 0.0676 |
| var@2.5x2.5x2.5mm | 66 | 62.1% | 0.577 | 0.2581 | 0.6635 | 0.2445 | +0.61 | 0.9463 | 0.0930 |
| 128@1.5x1.5x1.5mm | 30 | 36.7% | 0.339 | 0.4410 | 0.6572 | 0.4359 | +0.36 | 0.8373 | 0.1552 |

## Does a temperature transfer to an unseen centre?

Each family's predictions are calibrated with a temperature fitted on every *other*
family, then pooled. The shipped submission carries one scalar temperature that must
work on centres it never saw, so this is the closest free estimate of whether that
assumption holds.

| calibration | OOF log loss |
|---|---:|
| none | 0.3271 |
| global temperature (fitted on all OOF, in-sample) | 0.3186 |
| leave-one-family-out temperature | 0.3193 |

- transfer penalty = **+0.0006**

Read it like this: a penalty near zero means the calibration constant is a property of
the model, not of the centres it was fitted on, and the shipped temperature is safe. A
large positive penalty means the temperature is partly fitting centre composition, and
the leaderboard will not reproduce the OOF calibration gain.

### Temperature fitted without each family

Spread here is the diagnostic: a family whose exclusion moves the fitted temperature a
long way is a family that dominates the global calibration.

| family | n | T fitted excluding it |
|---|---:|---:|
| 256@2.5x2.5x2.5mm | 460 | 1.2197 |
| 128@3.9x3.9x3.9mm | 325 | 1.2565 |
| 128@2.3x2.3x2.3mm | 207 | 1.2969 |
| 128@2.4x2.4x2.4mm | 149 | 1.2799 |
| var@2.5x2.5x2.5mm | 66 | 1.2708 |
| 128@1.5x1.5x1.5mm | 30 | 1.2533 |

- range across reportable families: 1.2197 to 1.2969 (global 1.2614)

## Ceiling on per-centre prior correction

Give every family its own oracle logit offset, chosen with that family's own labels:

- pooled log loss with per-family oracle offsets: **0.3142**
- against uncalibrated 0.3271 — headroom **0.0129**

This is the absolute most that any per-centre prior correction could be worth, using
information no inference-time model can have. If the headroom is small, centre prior
shift is a second-order problem and the open question from row 0b can be closed. If it
is large, a shipped logit offset is worth designing — but it would have to be predicted
from the scan header alone, per sample, to stay inside the independence rule.

