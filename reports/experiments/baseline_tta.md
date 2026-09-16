# Evaluation: baseline_tta

- n = 1362, positive rate = 0.548
- **prior baseline log loss = 0.6884** (the floor to beat)

## Uncalibrated
- log loss **0.3239**
- AUROC 0.9306
- ECE 0.0370

## Calibration

| method | OOF log loss |
|---|---:|
| temperature | 0.3196 **<- chosen** |
| none | 0.3239 |
| isotonic | 0.3558 |

- chosen: **temperature** (temperature = 1.2457)
- ECE after calibration: 0.0269 (was 0.0370)
- AUROC after calibration: 0.9306 (calibration is monotonic, so this should be unchanged)

## Prediction clipping

| clip eps | OOF log loss |
|---|---:|
| 1e-06 | 0.3164 **<- best** |
| 0.0001 | 0.3164 |
| 0.001 | 0.3164 |
| 0.005 | 0.3164 |
| 0.01 | 0.3165 |
| 0.02 | 0.3178 |
| 0.05 | 0.3243 |

## Summary

- final OOF log loss: **0.3164**
- improvement from calibration + clipping: **0.0075**
- improvement over prior baseline: **0.3721**

## To ship this

```
python scripts/05_package_submission.py --config conf/<the config these weights were trained with> --temperature 1.2457 --fallback 0.5485 --clip-eps 1e-06
```

Copy the fold checkpoints into `submission_src/assets/` first.
