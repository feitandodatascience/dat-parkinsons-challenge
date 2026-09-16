# Evaluation: aug_rich

- n = 1362, positive rate = 0.548
- **prior baseline log loss = 0.6884** (the floor to beat)

## Training schedule

| fold | best epoch | last epoch | of | stopped early | lr at best / peak |
|---:|---:|---:|---:|---|---:|
| 0 | 18 | 29 | 30 | no | 0.35 |
| 1 | 21 | 29 | 30 | no | 0.21 |
| 2 | 18 | 29 | 30 | no | 0.35 |
| 3 | 15 | 29 | 30 | no | 0.50 |
| 4 | 21 | 29 | 30 | no | 0.21 |

- 5 folds trained.

## Uncalibrated
- log loss **0.3358**
- AUROC 0.9288
- ECE 0.0264

## Calibration

| method | OOF log loss |
|---|---:|
| none | 0.3358 **<- chosen** |
| temperature | 0.3363 |
| isotonic | 0.3624 |

- chosen: **none**
- ECE after calibration: 0.0264 (was 0.0264)
- AUROC after calibration: 0.9288 (calibration is monotonic, so this should be unchanged)

## Prediction clipping

| clip eps | OOF log loss |
|---|---:|
| 1e-06 | 0.3358 **<- best** |
| 0.0001 | 0.3358 |
| 0.001 | 0.3358 |
| 0.005 | 0.3358 |
| 0.01 | 0.3360 |
| 0.02 | 0.3371 |
| 0.05 | 0.3429 |

## Summary

- final OOF log loss: **0.3358**
- improvement from calibration + clipping: **0.0000**
- improvement over prior baseline: **0.3526**

## To ship this

> The comparison above chose **none**, but `submission_src/main.py` can only
> carry a scalar temperature, so the command below ships **temperature scaling**
> (OOF log loss 0.3340) rather than none.

```
python scripts/05_package_submission.py --config conf/<the config these weights were trained with> --temperature 1.1153 --fallback 0.5485 --clip-eps 1e-06
```

Copy the fold checkpoints into `submission_src/assets/` first.
